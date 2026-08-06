# The checkpoint you have is not the checkpoint you need

A model arrives as GGUF because that is what somebody had. The runtime you are
deploying on reads safetensors and names its modules the way MLX does. Nobody
wants to write a converter, so somebody writes forty lines that rename tensors
with a regular expression, and the result loads, runs, and is quietly wrong.

Three ways that happens, all of them present in the fixtures here:

The head dimension is not `hidden // heads`. The dense checkpoint in this
project carries 5120 hidden across 32 attention heads and still uses a head
dimension of 128, because it says so in a metadata field the converter did not
read. Divide instead of reading it and you build a `q_proj` of 5120 rows where
the weights have 4096.

A fused expert tensor is one tensor. In the mixture-of-experts checkpoint,
`blk.0.ffn_gate_exps.weight` holds all 128 experts stacked together. A mapping
that returns a single target name for it produces a model with one expert and
127 missing, which does not crash — it merely gets worse.

Dequantising is not free. The dense checkpoint is 14.3 GB of Q4_K and Q6_K. In
float16 it is 47 GB, and the machine it was going to run on has 36.

You are building `convertkit`: the tool that reads both sides, maps them, and
prices the job before anyone starts copying bytes.

## The fixtures

`_fixtures/gguf/tensor_index_llama.json` — every tensor of a real 40-layer
dense checkpoint: 363 names, shapes and ggml types, with the architecture
metadata. No weights, because a conversion planner does not need them.

`_fixtures/gguf/tensor_index_qwen3moe.json` — the same for a 48-layer
mixture-of-experts model with 128 experts, 8 of them active per token.

`_fixtures/safetensors/` — a two-shard export written by the reference
safetensors writer, with its `model.safetensors.index.json`, mixed F32, F16 and
BF16, and one damaged copy whose declared length no longer matches its shape.

`_fixtures/mlx/mlx_param_tree.json` — the parameter names MLX itself produces
for a llama-shaped model. This is the target convention, and in milestone 4
your mapping of the GGUF layer has to reproduce it exactly. Two independent
real sources agreeing is a stronger check than either one alone.

## The naming table

Per layer, `blk.N.X.weight` becomes `layers.N.Y.weight`:

    attn_q        self_attn.q_proj          ffn_gate      mlp.gate_proj
    attn_k        self_attn.k_proj          ffn_up        mlp.up_proj
    attn_v        self_attn.v_proj          ffn_down      mlp.down_proj
    attn_output   self_attn.o_proj          ffn_gate_inp  mlp.gate
    attn_norm     input_layernorm           attn_q_norm   self_attn.q_norm
    ffn_norm      post_attention_layernorm  attn_k_norm   self_attn.k_norm

Outside the layers: `token_embd.weight` becomes `embed_tokens.weight`,
`output_norm.weight` becomes `norm.weight`, `output.weight` becomes
`lm_head.weight`.

The three expert tensors fan out. `blk.N.ffn_gate_exps.weight` becomes
`layers.N.mlp.experts.E.gate_proj.weight` for every expert E, and the same for
`ffn_up_exps` and `ffn_down_exps`. With no expert count there is no correct
answer, and the right response is to say so rather than to guess.

Anything not in the table has no counterpart. Report it; do not invent one.

## Shapes

GGUF writes the fastest-moving dimension first, so the target shape is the
reverse of the stored one. Whether that reversal is right is checkable: the
metadata gives the hidden size, the head counts and the head dimension, and
`q_proj`, `k_proj` and `v_proj` have shapes those numbers determine. Both
fixtures use grouped-query attention — 32 heads over 8 key/value heads in one,
32 over 4 in the other — so k and v are narrower than q, and a converter that
assumes they match will not notice until generation goes wrong.

## Milestones

1. The safetensors header, parsed with `struct` and `json` and nothing else:
   offsets, dtypes, byte arithmetic, and a validator that names the damaged
   tensor.
2. The shard index: resolve every tensor to a file and a byte range, and catch
   an index that promises what the shards do not hold.
3. All 363 dense names and all 579 mixture-of-experts names, including the
   fan-out to 18,867 target tensors.
4. Shapes, grouped-query attention, the head dimension read rather than
   divided, and the cross-check against MLX's own naming.
5. The plan: bytes in, bytes out, the expansion factor, how many tensors need
   dequantising, and an output shard layout under a size limit.
6. `audit`, which returns the reasons not to run the conversion, and returns
   nothing at all when there are none.
