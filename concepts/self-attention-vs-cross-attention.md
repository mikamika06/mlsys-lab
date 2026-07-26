---
title: "What is self-attention vs cross-attention?"
description: "Self-attention vs cross-attention explained, with measured Q/K/V shapes, identical parameter counts, and the KV-cache-bytes table that shows why one grows every decode step and the other never does."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is self-attention vs cross-attention?

Self-attention and cross-attention run the exact same scaled-dot-product formula; the only
difference is which sequence supplies the keys and values. Self-attention draws Q, K and V
from one sequence, so every layer's score matrix is square and both K and V grow on every
decode step. Cross-attention draws Q from the decoder and K, V from a separate,
already-finished encoder output, so its score matrix is rectangular and its cache never grows
at all — measured below in shapes and cache bytes.

## How it works

Both mechanisms compute `softmax(QK^T / sqrt(d)) V` through identically-shaped projections
`Wq`, `Wk`, `Wv`, `Wo`, each `d_model x d_model`. Only the source of the inputs changes. In
self-attention one sequence `x` feeds all three: `Q = x Wq`, `K = x Wk`, `V = x Wv`. In
cross-attention two sequences feed them: a decoder sequence `x_dec` produces `Q = x_dec Wq`,
while a separate, already-computed encoder output `x_enc` produces `K = x_enc Wk` and
`V = x_enc Wv`. The weight matrices never notice which case they are in — they are
`d_model x d_model` either way, which is why the parameter count below is identical.

Shapes are where the two diverge. Self-attention over a sequence of length `L` produces an
`L x L` score matrix, every position against every position in the same sequence.
Cross-attention from a decoder of length `L` into an encoder of length `S` produces a
rectangular `L x S` matrix whose width is fixed the moment the encoder has run, independent of
how many decoder tokens follow. Masking follows the same split: self-attention needs a causal
mask so token `i` cannot see token `i+1`, while cross-attention needs no such mask, since the
entire encoder output already exists — at most a padding mask for variable-length inputs.

This is the mechanism inside every encoder-decoder architecture: the original Transformer's
translation decoder, T5, and Whisper's decoder each interleave causal self-attention over
tokens generated so far with cross-attention into an encoder output fixed for the rest of the
request. Decoder-only LLMs — GPT, Llama, and everything downstream — use *only*
self-attention; cross-attention appears nowhere in their forward pass, which is why the
distinction rarely comes up outside encoder-decoder and multimodal work.

The consequence that matters at serving time is the [KV cache](kv-cache.md). Self-attention's
`K` and `V` for token `t` are new the instant `t` is generated, so the cache appends a row every
decode step and keeps growing — growth [grouped query attention](grouped-query-attention.md)
shrinks per token, not stops. Cross-attention's `K` and `V` are computed once from the encoder
and only ever read afterward. A cache that never grows has nothing to page: block-table
indirection like [paged attention](paged-attention.md) exists for many growing self-attention
caches sharing a pool, a problem a fixed-size cross-attention cache never creates. Positional
encoding inherits the asymmetry too — [rotary embeddings](rope-embeddings.md) encode a position
*difference* along one sequence, an assumption that holds inside self-attention but not across
a decoder step and an unrelated encoder token index.

## Shapes and parameters, self-attention vs cross-attention

Fixed dimensions — batch 1, 8 heads, head dimension 64 (`d_model = 512`) — with a
self-attention sequence length of 2,048 against a cross-attention decoder length of 2,048
querying an encoder length of 512.

| quantity | self-attention (L=2,048) | cross-attention (L=2,048 → S=512) |
|---|---|---|
| Q shape (B,H,·,d) | (1, 8, 2048, 64) | (1, 8, 2048, 64) |
| K shape (B,H,·,d) | (1, 8, 2048, 64) | (1, 8, 512, 64) |
| V shape (B,H,·,d) | (1, 8, 2048, 64) | (1, 8, 512, 64) |
| score matrix shape | (1, 8, 2048, 2048) | (1, 8, 2048, 512) |
| Wq+Wk+Wv+Wo params | 1,048,576 | **1,048,576** |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
B, H, D = 1, 8, 64
D_MODEL = H * D
L, S = 2048, 512

def shapes(kind, B, H, D, L, S):
    Tq = L
    Tk = L if kind == "self" else S
    return (B, H, Tq, D), (B, H, Tk, D), (B, H, Tk, D), (B, H, Tq, Tk)

def attn_params(d_model):
    return 4 * d_model * d_model

for kind in ("self", "cross"):
    q, k, v, sc = shapes(kind, B, H, D, L, S)
    print(f"{kind}: Q={q} K={k} V={v} scores={sc}")

print("params (Wq+Wk+Wv+Wo):", attn_params(D_MODEL))
PY
```

Q keeps the same shape in both rows because a query always comes from the decoder side — that
never changes. K and V shrink from `2048` to `512` the moment they come from the encoder
instead of the decoder's own history, which is exactly why the score matrix goes from
`2048 x 2048` to `2048 x 512`: cross-attention's score matrix is bounded by the encoder length,
not the ever-growing decoder length. The parameter row does not move at all — `1,048,576`
either way — because `d_model` alone fixes every projection's shape.

## Practise it

```bash
mlsys grade rwa-implement-scaled-dot-product-attention-semantics
```

[That task](../tasks/rwa-implement-scaled-dot-product-attention-semantics/task.md) gates
`scaled_dot_product_attention(Q, K, V, mask, causal)` against a NumPy reference on
`max_abs_err`. Its signature already takes `Q` shaped `(B, H, T_q, d_k)` and `K`/`V` shaped
`(B, H, T_k, d_k)` — pass `T_q == T_k` for self-attention, `T_q != T_k` for cross-attention,
same function, no branch. That is the real lesson: one operator, a different `T_k`.

More of the same mechanism, increasing scope:
[the S, P, O stages of the same formula](../tasks/llm-round-trip-attention-stages-s-p-o/task.md),
[decomposing SDPA into ONNX primitives across a rectangular `L_q != L_k`
case](../tasks/rwc-decompose-sdpa-into-onnx-primitives-and-match-fused-attention/task.md),
[GQA/MQA key-value broadcast checked against MHA and a `T_q != T_k`
case](../tasks/rwc-gqa-mqa-attention-with-kv-head-broadcast/task.md),
[counting the attention FLOPs the shape table above feeds
into](../tasks/rwb-attention-flop-count/task.md),
[multi-head parameter and FLOP counting against an equal-`d` single
head](../tasks/llm-mha-flops-params-vs-equal-d-single-head/task.md), and
[mapping multi-head attention onto a GPU's thread
hierarchy](../tasks/gpu-map-multi-head-attention-onto-the-thread-hierarchy/task.md).

## KV-cache bytes during incremental decoding

The real asymmetry shows up once decoding starts. Per layer, fp16, 8 KV heads, head dimension
64: self-attention's cache grows with the decode step `t`; cross-attention's cache is computed
once against the fixed encoder length `S=512` and stays there for every `t`.

| decode step t | self-attention K+V bytes | cross-attention K+V bytes |
|---|---|---|
| 1 | 2,048 | 1,048,576 |
| 512 | 1,048,576 | 1,048,576 |
| 1,024 | 2,097,152 | 1,048,576 |
| 2,048 | 4,194,304 | 1,048,576 |
| 4,096 | 8,388,608 | 1,048,576 |
| 8,192 | **16,777,216** | 1,048,576 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

H, D, S = 8, 64, 512
ITEMSIZE = np.dtype("float16").itemsize

def kv_cache_bytes_per_layer(H, D, T, itemsize):
    return 2 * H * D * T * itemsize

for t in (1, 512, 1024, 2048, 4096, 8192):
    self_bytes = kv_cache_bytes_per_layer(H, D, t, ITEMSIZE)
    cross_bytes = kv_cache_bytes_per_layer(H, D, S, ITEMSIZE)
    print(f"t={t} self_bytes={self_bytes} cross_bytes={cross_bytes}")
PY
```

Self-attention's cache passes cross-attention's fixed `1,048,576` bytes at exactly `t=512` —
the point where the decoder has generated as many tokens as the encoder had — and keeps
climbing linearly past it, reaching `16,777,216` bytes by `t=8,192` while cross-attention's
number has not moved once in any row. That row is per layer and ignores the layer-count
multiplier this repo's [KV cache](kv-cache.md) page applies in full; the ratio between the two
columns is unaffected by that multiplier, since both scale by the same layer count.

## Common mistakes

- **Giving cross-attention a causal mask.** The encoder output already exists in full — there
  is no future position to hide. Causal masking belongs to self-attention's decoder side;
  cross-attention needs at most a padding mask.
- **Assuming the parameter count differs.** It never does: `Wq`, `Wk`, `Wv`, `Wo` are
  `d_model x d_model` regardless of source, so switching a layer to cross-attention changes
  activation shapes and cache growth, not weight count — `1,048,576` either way above.
- **Sizing a cross-attention cache like a self-attention one.** A cross-attention cache is a
  one-time, `S`-length allocation; paging it into growable blocks the way
  [paged attention](paged-attention.md) pages a self-attention cache spends effort on a cache
  that was never going to grow.
- **Reusing self-attention's positional scheme unchanged.** [RoPE](rope-embeddings.md) encodes
  a position *difference* along one sequence; a decoder step and an encoder token index are not
  positions on the same axis, so applying the identical rotation across both encodes a
  difference that is not meaningful.
- **Expecting cross-attention in a decoder-only model.** GPT- and Llama-style models have no
  cross-attention layer anywhere; KV-cache-growth intuition built on them does not carry over
  the day cross-attention is added.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists
attention and KV-cache as 124 tasks in this bank with some overlap elsewhere:

- **[Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)**
  — a dedicated self-attention problem among its six confirmed items; no cross-attention
  problem and no KV-cache-growth comparison anywhere.
- **[LeetGPU — challenge set](https://leetgpu.com/challenges)** — one `multi-head-attention`
  kernel among ~90, self-attention only, on a CPU-emulated free tier, no encoder-decoder or
  cache-growth variant.
- **[Stanford CS336 — Assignment 2 (Systems)](https://github.com/stanford-cs336/assignment2-systems)**
  — a real FlashAttention-2 forward/backward kernel, pytest-checked; self-attention throughout,
  since the course targets a decoder-only architecture.
- **[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)** — the
  production reference kernel library, MQA/GQA and varlen included; read-only, the canonical
  place to see a real rectangular cross-attention path once the exercises above stop being
  enough.
- Nothing surveyed grades the self-vs-cross distinction itself or measures the KV-cache-growth
  asymmetry this page's second table isolates.

## References

1. Vaswani, A. et al., *Attention Is All You Need*, NeurIPS 2017 — the original
   encoder-decoder architecture defining both mechanisms.
   https://arxiv.org/abs/1706.03762
2. Raffel, C. et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text
   Transformer* (T5), JMLR 2020 — a concrete encoder-decoder model whose decoder blocks
   interleave causal self-attention with cross-attention into a fixed encoder output.
   https://arxiv.org/abs/1910.10683
3. PyTorch documentation, `torch.nn.functional.scaled_dot_product_attention` — the same
   operator accepting `query`/`key`/`value` of independent sequence lengths, the API-level form
   of the `T_q != T_k` case this page measures.
   https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
