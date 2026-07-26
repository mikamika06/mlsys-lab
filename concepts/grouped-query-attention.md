---
title: "What is grouped query attention?"
description: "Grouped query attention explained, with a measured KV-cache-bytes-vs-groups table — and the exact context length where the cache outweighs the weights — you can reproduce, plus a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is grouped query attention?

Grouped query attention splits a transformer's query heads into groups that each
share one key/value projection, instead of every head keeping its own. The saving
is not cosmetic: for a 32-head, 32-layer, fp16 model at a 128K-token context,
dropping from 32 KV heads to the usual 8 cuts the cache from 64 GiB to 16 GiB
without touching a single weight. Below, that shrinkage is measured across every
group count from 32 down to 1, together with the context length at which the
cache — not the weights — becomes the thing that stops fitting.

## How it works

During generation a transformer does not recompute attention over the whole
prefix at every step — it keeps every past token's key and value vectors
resident, one pair per layer per head, so the next token's query can attend to
them directly. That store is the KV cache, and its size is set entirely by how
many separate key/value projections the model keeps: full multi-head attention
(MHA) gives every one of its `h` query heads its own K/V pair, so the cache grows
linearly with head count. Grouped query attention (GQA) breaks that link: heads
are split into `g` groups, and every head inside a group reads the same K/V pair,
so the cache scales with `g` instead of `h` while the query side — and therefore
the model's representational width — is untouched.

The reason this matters more than an equivalent saving in the weights would is
that decoding one token is memory-bandwidth bound, not compute bound: the
accelerator must stream the entire KV cache back off HBM for every single
generated token, the same "bytes moved decide the cost, not FLOPs" fact that
makes [memory coalescing](memory-coalescing.md) matter on the read side of a
kernel and [continuous batching](continuous-batching.md) matter on the
scheduling side — both exist because the cache, not the arithmetic, is what a
serving system actually rations.

GQA is usually not trained from scratch; production checkpoints are converted
from an existing MHA model by mean-pooling each group's original per-head K/V
weights into one shared pair, then briefly fine-tuning to recover quality —
cheaper than a fresh pretraining run for the same KV-bandwidth win. Two further
levers stack on top of the group count: a narrower storage format
([bfloat16 vs float16](bfloat16-vs-float16.md) is one axis,
[int4/int8 KV quantization](integer-quantization-ranges.md) another) shrinks
*bytes per element*, while GQA shrinks *element count* — combine both and the
savings multiply. The same "nominal size is not real size" instinct drives
[GGUF vs safetensors](gguf-vs-safetensors.md)' block-quantization accounting,
and the same "spend memory or spend compute" trade shows up on the training side
as [gradient checkpointing](gradient-checkpointing.md), applied to activations
instead of the KV cache.

## Multi-query attention: the one-group floor

Multi-query attention (MQA) is grouped query attention with `g` fixed at 1: every
query head in every layer attends against the same single shared key/value pair
— the cheapest KV cache the formula can produce for a given `head_dim` and layer
count. In the table below it is the 16,384-bytes-per-token row, 32× smaller than
MHA's 524,288, but also the most aggressive compression: 32 heads now agree on
one shared view of the past, the largest single source of the quality gap GQA
was introduced to close. MQA predates GQA by roughly four years and is still
shipped where bandwidth matters more than the quality it costs; GQA is best read
as "MQA generalized to a tunable group count," not a different mechanism, which
is why this page treats both with one formula and table rather than a second
page.

## KV cache bytes against KV groups, and where they cross the weights

The shape held fixed throughout: 32 query heads, `head_dim` 128 (so `d_model` =
4,096), 32 layers, fp16 — Llama-3-8B's actual attention configuration. Only the
KV group count is varied. For each group count the script allocates the real K/V
arrays to get bytes/token, multiplies by a 131,072-token (128K) context for the
total, and separately allocates the model's own weight matrices (`Wq`, `Wk`,
`Wv`, `Wo`, both MLP matrices, one tied embedding at a stand-in 32K vocabulary
and generic 4× MLP width, since Llama-3's real FFN and vocabulary are wider and
would only dilute the point) to find the context length where cache bytes equal
weight bytes.

| KV groups | notes | KV bytes/token | KV total @128K tokens (GiB) | model weight bytes, fp16 (GiB) | KV ÷ weights @128K | crossover (tokens) |
|---:|---|---:|---:|---:|---:|---:|
| 32 | MHA | 524,288 | 64.00 | 12.24 | 5.23× | 25,076 |
| 16 | | 262,144 | 32.00 | 11.24 | 2.85× | 46,056 |
| 8 | GQA, Llama-3-8B | 131,072 | 16.00 | 10.74 | 1.49× | 88,016 |
| 4 | | 65,536 | 8.00 | 10.49 | 0.76× | 171,936 |
| 2 | | 32,768 | 4.00 | 10.37 | 0.39× | 339,776 |
| 1 | MQA | 16,384 | 2.00 | 10.31 | 0.19× | 675,456 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

layers, num_heads, head_dim = 32, 32, 128
d_model = num_heads * head_dim
d_ff = 4 * d_model
vocab = 32000
dtype = np.float16
CTX = 131072

def kv_bytes_per_token(groups):
    k = np.empty((layers, groups, head_dim), dtype=dtype)
    v = np.empty_like(k)
    return k.nbytes + v.nbytes

def weight_bytes(groups):
    d_kv = groups * head_dim
    wq = np.empty((d_model, d_model), dtype=dtype)
    wk = np.empty((d_model, d_kv), dtype=dtype)
    wv = np.empty((d_model, d_kv), dtype=dtype)
    wo = np.empty((d_model, d_model), dtype=dtype)
    w1 = np.empty((d_model, d_ff), dtype=dtype)
    w2 = np.empty((d_ff, d_model), dtype=dtype)
    per_layer = wq.nbytes + wk.nbytes + wv.nbytes + wo.nbytes + w1.nbytes + w2.nbytes
    emb = np.empty((vocab, d_model), dtype=dtype)
    return per_layer * layers + emb.nbytes

for groups in (32, 16, 8, 4, 2, 1):
    per_tok = kv_bytes_per_token(groups)
    total = per_tok * CTX
    wb = weight_bytes(groups)
    crossover = wb / per_tok
    ratio_128k = total / wb
    print(f"groups={groups:>2} bytes_per_token={per_tok:>7} "
          f"total_128k_GiB={total/2**30:.2f} weight_GiB={wb/2**30:.2f} "
          f"ratio_128k={ratio_128k:.2f} crossover_tokens={crossover:.0f}")
PY
```

Two things the table says that "GQA saves memory" does not. First, the crossover
is not a far-future hypothetical: at 32 heads (plain MHA) the cache already
outweighs the weights past 25,076 tokens — inside most current serving context
windows — while GQA-8 pushes that to 88,016 and MQA to 675,456, a length few
deployed systems reach. Second, the weight side barely moves across the whole
sweep (12.24 GiB down to 10.31 GiB, a 16% drop) because `Wk`/`Wv` are a small
slice of parameters next to `Wq`, `Wo`, and the MLP, while the KV side moves
32×: shrinking the group count is a near-free lever on weight budget and the
dominant lever on serving memory budget, which is why it is the industry
default rather than a niche optimization.

## Practise it

```bash
mlsys grade llm-kv-cache-bytes-mha-vs-gqa-vs-mqa
```

[That task](../tasks/llm-kv-cache-bytes-mha-vs-gqa-vs-mqa/task.md) gates a
`kv_cache_bytes` function on two metrics, `gqa_mha_ratio == 1.0` and
`mqa_mha_ratio == 1.0`, checked against a NumPy oracle. The shipped starter
raises `NotImplementedError`, so it fails immediately. The tempting wrong fix
mirrors the starter's own warning docstring: computing the GQA branch with
`num_heads` instead of `groups` still returns a dict of the right shape and
dtype, so it passes any shape check, but it silently reproduces MHA's byte count
— `gqa_mha_ratio` only hits `1.0` when `g` happens to equal `h`, and the gate
fails for every real GQA config where it doesn't.

More of the same shape, in increasing difficulty:
[classify a checkpoint as MHA, GQA, or MQA from its tensor shapes alone](../tasks/llm-classify-checkpoint-mha-gqa-mqa-from-shapes/task.md)
(no code), [classify a config and compute its `n_rep`](../tasks/rwa-classify-a-config-as-mha-gqa-mq-and-compute-n-rep/task.md),
[implement MQA's single-KV-head broadcast](../tasks/rwa-implement-mqa-single-kv-head-broadcast/task.md),
[implement GQA head-expansion inside scaled-dot-product attention](../tasks/rwa-implement-gqa-head-expansion-sdpa/task.md),
and [measure GQA's effect on decode arithmetic intensity](../tasks/llm-gqa-effect-on-decode-arithmetic-intensity/task.md).

## Common mistakes

- **Reading "8 KV heads" as "8× smaller model."** GQA-8 shrinks the KV cache 4×
  per token relative to MHA, but total weight bytes drop only from 12.24 GiB to
  10.74 GiB — 12% — because `Wk`/`Wv` are a small fraction of parameters next to
  `Wq`, `Wo`, and the MLP.
- **Assuming the crossover only matters at extreme context lengths.** MHA's
  cache already outweighs its own weights past 25,076 tokens, well inside a
  single modern context window.
- **Treating GQA and quantized KV cache as competing choices.** They compose:
  cutting groups 32→8 is a 4× reduction on its own; adding
  [int4/int8 KV quantization](integer-quantization-ranges.md) is a further,
  independent reduction in bytes per element, and the two multiply rather than
  substitute.
- **Confusing MQA with "GQA that failed."** MQA is `g = 1`, the correct floor of
  the same formula, not a deprecated mechanism — it is the right choice whenever
  the quality it costs is smaller than the bandwidth it buys back.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[LeetGPU — challenge set](https://leetgpu.com/challenges)** has a dedicated
  `grouped-query-attention` challenge among its attention/KV-cache set, graded
  free on a CPU-emulated GPU. It checks a kernel's numerical output, not
  KV-cache byte accounting.
- **[Deep-ML — Attention Is All You Need collection](https://www.deep-ml.com/collections/Attention%20Is%20All%20You%20Need)**
  has one confirmed GQA problem among six attention ones, browser IDE with
  instant pass/fail; freemium, some content paywalled.
- **[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)**
  is the production reference — its CUDA/CUTLASS kernels natively support
  MQA/GQA, varlen, and paged KV — but it is read-only, nothing to submit
  against.
- Nothing found pairs the group-count math with a byte-level KV-cache crossover
  calculation in a graded exercise; that gap, plus the cache-management tasks
  (block tables, paged allocation, eviction) this bank also carries, is where
  the 124-task attention-and-KV-cache area here fills empty space rather than
  competing in a crowded one.

## References

1. Ainslie, J. et al., *GQA: Training Generalized Multi-Query Transformer Models
   from Multi-Head Checkpoints*, 2023. https://arxiv.org/abs/2305.13245
2. Shazeer, N., *Fast Transformer Decoding: One Write-Head is All You Need*,
   2019. https://arxiv.org/abs/1911.02150
3. Meta Llama 3 model config — `num_attention_heads=32`,
   `num_key_value_heads=8`, `num_hidden_layers=32`, `head_dim=128`.
   https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
