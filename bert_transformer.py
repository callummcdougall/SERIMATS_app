# %%

import os
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookedRootModule
from transformer_lens.components import HookPoint
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformer_lens.components import LayerNorm, Embed, PosEmbed, Attention, MLP, Unembed, TransformerBlock
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from collections import OrderedDict
from typing import Tuple, List, Optional, Union, Iterable, Literal, cast, Callable
from jaxtyping import Float, Int
from rich import print as rprint
import transformers
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import logging

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

if MAIN:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%

class BERTBlock(nn.Module):
    '''
    This is the first main difference from TransformerLens, because BERTBlock swaps the order of layernorms. 
    '''
    def __init__(self, cfg: HookedTransformerConfig, block_index: int):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
        self.attn = Attention(cfg, layer_id=block_index)
        self.mlp = MLP(cfg)

        self.hook_q_input = HookPoint()  # [batch, pos, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(self, resid_pre: Tensor) -> Tensor:

        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

        query_input = self.hook_q_input(resid_pre)
        key_input = self.hook_k_input(resid_pre)
        value_input = self.hook_v_input(resid_pre)

        # attn, then adding, then ln1
        attn_out = self.hook_attn_out(self.attn(
            query_input, key_input, value_input
        ))  # [batch, pos, d_model]
        resid_mid = self.hook_resid_mid(
            self.ln1(resid_pre + attn_out)
        )

        # mlp, then adding, then ln2
        mlp_out = self.hook_mlp_out(
            self.mlp(resid_mid)
        )  # [batch, pos, d_model]
        resid_post = self.hook_resid_post(
            self.ln2(resid_mid + mlp_out)
        )  # [batch, pos, d_model]

        return resid_post



class BERT(HookedTransformer):
    '''
    Now we assemble the BERTBlocks together into BERT (and add extra bits like linear & gelu at the end).

    Also define some useful methods for (1) adding permanent hooks for attn masking, and (2) loading in weights
    in a non-agressively-stupid way.
    '''
    def __init__(
        self, 
        cfg: HookedTransformerConfig, 
        tokenizer: Optional[BertTokenizerFast] = None,
        weights: Optional[BertForMaskedLM] = None,
    ):
        # The HookedTransformer init method does quite a few of the things we need, so we call it first
        # (not sure exactly what here we're allowed to ditch)
        super().__init__(cfg)
        delattr(self, "ln_final")
        self.ln_first = LayerNorm(cfg)

        self.tokenizer = get_bert_tokenizer() if (tokenizer is None) else tokenizer

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        # Extremely hacky way of getting the token type embedding (which is always vector of zeros)
        token_type_cfg = {**cfg.__dict__, "d_vocab": 2}
        self.token_type_embed = Embed(token_type_cfg)

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        self.hook_tokens = HookPoint() # [batch, pos]

        self.blocks: Iterable[BERTBlock] = nn.ModuleList([
            BERTBlock(self.cfg, block_index)
            for block_index in range(self.cfg.n_layers)
        ])

        # linear -> gelu -> layernorm -> unembed
        # (rather than e.g. GPT2 which just has final ln then unembed here)
        self.block_final = nn.Sequential(OrderedDict([
            ("linear_final", nn.Linear(cfg.d_model, cfg.d_model)),
            ("gelu", nn.GELU()),
        ]))
        self.ln_final = LayerNorm(cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

        # Add permanent hooks (never get removed ideally)
        self.add_perma_hooks_for_masking_PAD()

        # Load in weights
        if weights is not None:
            self.load_weights(weights)

        self = self.to(cfg.device)


    def add_perma_hooks_for_masking_PAD(self) -> None:
        '''
        This function adds permanent hooks for masking the [PAD] tokens in BERT.

        This diagram explains how it works conceptually: tinyurl.com/mr2u9erf
        '''
        # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
        def cache_padding_tokens_mask(
            tokens: Int[Tensor, "batch seq"],
            hook: HookPoint,
        ) -> None:
            batch, seq_len = tokens.shape
            hook.ctx["padding_tokens_mask"] = einops.repeat(
                tokens == self.tokenizer.pad_token_id,
                "batch seq_K -> batch head seq_Q seq_K",
                head=self.cfg.n_heads, seq_Q=seq_len
            )

        # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
        # If this attention layer is the last one, then clear the information from the `hook_tokens` context
        def apply_padding_tokens_mask(
            attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
            hook: HookPoint,
        ) -> None:
            attn_scores.masked_fill_(self.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1.0e5)
            if hook.layer() == self.cfg.n_layers - 1:
                del self.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

        # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
        self.add_perma_hook(name="hook_tokens", hook=cache_padding_tokens_mask)
        self.add_perma_hook(name=lambda x: x.endswith("attn_scores"), hook=apply_padding_tokens_mask)

    def forward(
        self,
        input: Union[str, List[str], Int[Tensor, "batch pos"]],
        return_type: Optional[Literal["logits", "loss"]] = "logits",
    ) -> Tensor:
        '''
        Tokenization uses BERT's special tokenizer.

        Input mask is inferred from the tokenizer's pad token id.
        '''
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            tokens = self.to_tokens(input, prepend_bos=False)
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
        assert isinstance(tokens, Tensor)

        tokens = self.hook_tokens(tokens)  # [batch, pos]
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens))  # [batch, pos, d_model]
        token_type_embed = self.token_type_embed(t.zeros_like(tokens))  # [batch, pos, d_model]
        residual = self.ln_first(embed + pos_embed + token_type_embed)  # [batch, pos, d_model]

        for i, block in enumerate(self.blocks):
            residual = block(residual)  # [batch, pos, d_model]
        
        residual = self.block_final(residual)  # [batch, pos, d_model]
        residual = self.ln_final(residual)  # [batch, pos, d_model]

        if return_type is None:
            return None
        else:
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            elif return_type == "loss":
                return self.loss_fn(logits, tokens)
            

    def loss_fn(
        self, 
        logits: Float[Tensor, "batch posn d_vocab"], 
	    tokens: Int[Tensor, "batch posn"]
    ) -> Float[Tensor, "batch"]:
        '''
        Needs to be different to the default function, because we only care about loss on [MASK] tokens.
        '''
        is_MASK_token = (tokens == self.tokenizer.mask_token_id).flatten()

        ignore_index = -100
        masked_tokens = t.where(is_MASK_token, tokens, t.tensor(ignore_index)).flatten()

        logits = einops.rearrange(logits, "batch posn d_vocab -> (batch posn) d_vocab")
        
        loss = F.cross_entropy(logits, masked_tokens, ignore_index=ignore_index)

        return loss
    

    def load_weights(self, bert_loaded: BertForMaskedLM) -> "BERT":
        named_params_loaded = {}
        for k, v in bert_loaded.named_parameters():
            mynames, tensorfns = convert_bertloaded_to_bert_name(k)
            for name, tensorfn in zip(mynames, tensorfns):
                named_params_loaded[name] = tensorfn(v)
        for name, param in self.named_parameters():
            assert param.shape == named_params_loaded[name].shape, f"shape mismatch for {name}: {param.shape} vs {named_params_loaded[name].shape}"
            param.data = named_params_loaded[name]
        return self


def get_bert_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.bos_token = "[CLS]"
    return tokenizer


def get_pretrained_bert() -> BertForMaskedLM:
    """Load the HuggingFace BERT.
    Supresses the spurious warning about some weights not being used.
    """
    logger = logging.getLogger("transformers.modeling_utils")
    was_disabled = logger.disabled
    logger.disabled = True
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    logger.disabled = was_disabled
    return cast(BertForMaskedLM, bert)


def convert_bertloaded_to_bert_name(name: str) -> Tuple[List[str], List[Callable]]:
    '''
    Takes name of a bert_loaded param, and returns (mynames, tensorfns)
    
        mynames: List[str]
            List of names of my model's tensors that correspond to this param
            This is always length 1, except for W_E (because bert_loaded doesn't count this twice, but we do)

        tensorfns: List[Callable]
            Function that takes a tensor and returns the tensor that should be used in my model
            This is sometimes just identity or transposition, sometimes there's also some einops rearranging
    '''
    # default case: identity
    id = lambda x: x
    tr = lambda x: x.transpose(0, 1)

    if name == "bert.embeddings.word_embeddings.weight":
        mynames = ["embed.W_E", "unembed.W_U"]
        tensorfns = [id, tr]
    elif name == "bert.embeddings.token_type_embeddings.weight":
        mynames = ["token_type_embed.W_E"]
        tensorfns = [id]
    elif name == "bert.embeddings.position_embeddings.weight":
        mynames = ["pos_embed.W_pos"]
        tensorfns = [id]
    elif "embeddings.LayerNorm" in name:
        mynames = ["ln_first.w"] if name.endswith("weight") else ["ln_first.b"]
        tensorfns = [id]
    elif "bert.encoder.layer" in name:
        newname = (name
            # first, deal with the big ones
            .replace("bert.encoder.layer", "blocks")
            # next, deal with keys / queries / values / outputs
            .replace("attention.output.dense.weight", "attn.W_O")
            .replace("attention.output.dense.bias", "attn.b_O")
            .replace("attention.self.query.weight", "attn.W_Q")
            .replace("attention.self.query.bias", "attn.b_Q")
            .replace("attention.self.key.weight", "attn.W_K")
            .replace("attention.self.key.bias", "attn.b_K")
            .replace("attention.self.value.weight", "attn.W_V")
            .replace("attention.self.value.bias", "attn.b_V")
            # next, layernorms
            .replace("attention.output.LayerNorm.weight", "ln1.w")
            .replace("attention.output.LayerNorm.bias", "ln1.b")
            .replace("output.LayerNorm.weight", "ln2.w")
            .replace("output.LayerNorm.bias", "ln2.b")
            # finally, MLPs
            .replace("intermediate.dense.weight", "mlp.W_in")
            .replace("intermediate.dense.bias", "mlp.b_in")
            .replace("output.dense.weight", "mlp.W_out")
            .replace("output.dense.bias", "mlp.b_out")
        )
        if ("mlp.b" in newname) or ("attn.b_O" in newname) or (".ln" in newname):
            tensorfn = id
        elif "mlp.W" in newname:
            tensorfn = tr
        elif "attn.W_O" in newname:
            tensorfn = lambda w: einops.rearrange(w, "d_model (nheads d_head) -> nheads d_head d_model", nheads=12)
        elif "attn.W" in newname:
            tensorfn = lambda w: einops.rearrange(w, "(nheads d_head) d_model -> nheads d_model d_head", nheads=12)
        elif "attn.b" in newname:
            tensorfn = lambda b: einops.rearrange(b, "(nheads d_head) -> nheads d_head", nheads=12)
        mynames = [newname]
        tensorfns = [tensorfn]
    else:
        # Note we don't do trace for this linear layer, cause it's left-mult like nn.Linear (not like TL)
        # so all of these are just id
        tensorfns = [id]
        mynames = [{
            "cls.predictions.bias": "unembed.b_U",
            "cls.predictions.transform.dense.weight": "block_final.linear_final.weight",
            "cls.predictions.transform.dense.bias": "block_final.linear_final.bias",
            "cls.predictions.transform.LayerNorm.weight": "ln_final.w",
            "cls.predictions.transform.LayerNorm.bias": "ln_final.b",
        }[name]]
    return mynames, tensorfns


# %%

if MAIN:
    tokenizer = get_bert_tokenizer()
    assert tokenizer.tokenize("John and Mary", add_special_tokens=True) == ["[CLS]", "John", "and", "Mary", "[SEP]"]

    cfg = HookedTransformerConfig(
        d_model = 768,
        eps = 1e-12,
        d_vocab = 28996,
        n_ctx = 512,
        d_head = 64,
        d_mlp = 3072,
        n_heads = 12,
        n_layers = 12,
        act_fn = "gelu",
        attention_dir = "bidirectional",
        use_hook_tokens=True, # technically don't need this because my architecture BERT assumes it
        device = device,
    )

    bert_loaded = get_pretrained_bert()
    bert = BERT(cfg, tokenizer, weights=bert_loaded)
    # tab = Table("MyName", "BERT_Loaded_Shape", "BERT_Shape")

# %%

# Running it on these prompts; within numerical error.

if MAIN:
    input_str = "John and Mary [MASK] to the shops"
    output = bert(input_str)

    input = tokenizer(input_str, return_tensors="pt")
    output_loaded = bert_loaded(input.input_ids, input.attention_mask, input.token_type_ids).logits.to(device)

    avg_error = (output - output_loaded).abs().mean()
    print(f"Average error: {avg_error:.3e}")
    t.testing.assert_close(output, output_loaded, rtol=1e-4, atol=1e-4)

# %%

def test_bert(
    bert: BERT,
    bert_loaded: BertForMaskedLM, 
    prompt="The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK].",
    k=3,
):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    logprobs_actual = bert(input_ids).to(device)[input_ids == tokenizer.mask_token_id].log_softmax(-1)
    top_logprobs, top_tokens_actual = logprobs_actual.topk(k, dim=-1)
    top_probs = top_logprobs.squeeze().exp().tolist()
    top_tokens_actual = [(tokenizer.decode(tok), prob) for (prob, tok) in zip(top_probs, top_tokens_actual.squeeze())]

    logprobs_expected = bert_loaded(input_ids).logits.log_softmax(-1).to(device)[input_ids == tokenizer.mask_token_id]

    print(f"Prompt: {prompt!r}\n")
    avg_error = (logprobs_actual - logprobs_expected).abs().mean()
    print(f"Average logit difference: {avg_error:.3e}\n")
    print(f"Top responses from my model:")
    rprint(top_tokens_actual)
    t.testing.assert_close(logprobs_actual, logprobs_expected, rtol=1e-4, atol=1e-4)
    print("Got expected responses!")

if MAIN:
    test_bert(bert, bert_loaded)

# %%

def test_bert_with_padding(
    bert: BERT,
    bert_loaded: BertForMaskedLM, 
    prompts=[
        "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK].",
        "The President of the United States is [MASK].",
    ],
    k=3
):
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
    attn_mask = tokenizer(prompts, return_tensors="pt", padding=True)["attention_mask"]

    logits_actual = bert(input_ids).to(device)[input_ids == tokenizer.mask_token_id]
    top_tokens_actual = logits_actual.topk(k, dim=-1).indices
    top_tokens_actual = [[tokenizer.decode(tok) for tok in seq] for seq in top_tokens_actual]

    logits_expected = bert_loaded(input_ids, attn_mask).logits.to(device)[input_ids == tokenizer.mask_token_id]
    top_tokens_expected = logits_expected.topk(k, dim=-1).indices
    top_tokens_expected = [[tokenizer.decode(tok) for tok in seq] for seq in top_tokens_expected]

    print(f"Prompts: {prompts!r}\n")
    avg_error = (logits_actual - logits_expected).abs().mean(-1).tolist()
    print(f"Average logit difference: {avg_error[0]:.3e}, {avg_error[1]:.3e}\n")
    print(f"Top responses from pretrained model: {top_tokens_expected}")
    print(f"Top responses from my model:         {top_tokens_actual}\n")
    assert top_tokens_expected == top_tokens_actual
    print("Got expected responses!")

if MAIN:
    test_bert_with_padding(bert, bert_loaded)

# %%

# Test whether BERT can actually solve the IOI task!
# result - yes, it definitely can

if MAIN:
    prompt = "When John and Mary went to the store, [MASK] gave a drink to Mary"
    test_bert(bert, bert_loaded, prompt=prompt)

# %%

# test cache, does it work same as hooks when getting attn probs?

def test_caching_vs_hooks(bert: BERT):
    prompt = "one two three"
    _, cache = bert.run_with_cache(prompt, return_type=None, names_filter=lambda x: x.endswith("pattern"))
    cache_from_hooks = {}
    def hook_fn(x, hook: HookPoint):
        print(f"Hook is running at {hook.name!r}")
        cache_from_hooks[hook.name] = x
    bert.run_with_hooks(
        prompt,
        return_type = None,
        fwd_hooks = [(lambda x: x.endswith("pattern"), hook_fn)],
    )
    assert len(cache) == len(cache_from_hooks)
    for k, v in cache_from_hooks.items():
        assert k in cache
        t.testing.assert_close(v, cache[k])
        assert v.min() >= 0
        assert v.max() <= 1
    print("All tests passed!")


if MAIN:
    test_caching_vs_hooks(bert)


# %%
