---
title: "What is kv cache?"
description: "KV cache explained, with a measured bytes-per-token table across real model shapes, the context length at which it outgrows the weights, and a graded exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is kv cache?

KV cache is the per-token store of attention key and value vectors that a
decoder-only transformer keeps so it never recomputes them on a later step.
Skipping it does not just cost time: naively rebuilding attention from
scratch at every one of 8,192 decode steps costs 2,730.8x the attention
FLOPs of reading the cache back, measured below. What follows turns that
trade into exact bytes per token, then into the context length at which the
cache outgrows the weights sitting beside it.

## KV cache explained

A decoder-only transformer produces one token per step, and every step's
causal self-attention needs the key and value vectors of every token
generated so far, not only the newest one. Recomputing those from scratch
each step means re-running every earlier token through each layer's
key/value projections again — step $t$ redoes $t-1$ tokens' worth of
projection, so cost grows with the square of tokens generated. The KV
cache avoids exactly that: each layer keeps the key and value vectors it
has already computed, appends the new token's the moment they exist, and
every later step reads the growing cache instead of rebuilding it.

What the cache costs is fixed by four numbers: layer count, the number of
*key/value* heads specifically (grouped-query and multi-query attention
deliberately shrink this, not the query-head count), head dimension, and
storage dtype. Dtype follows the accounting already measured for weights:
[bytes-per-weight, actual vs nominal](gguf-vs-safetensors.md) and
[bfloat16 vs float16](bfloat16-vs-float16.md) apply unchanged, and a cache
stored in [int8 instead of fp16](integer-quantization-ranges.md) halves
its footprint at the cost of the same quantization error those pages
measure for weights.

The KV cache mirrors [gradient checkpointing](gradient-checkpointing.md):
checkpointing throws activations away and pays compute to reconstruct
them; the cache pays memory up front so nothing downstream is
reconstructed — the same memory-for-compute lever, pointed opposite ways
for training's backward pass against inference's decode loop.

At serving scale the cache becomes hundreds of growing arrays, one per
active request, which is why engines page it into fixed-size blocks
rather than pre-allocate a worst-case buffer per sequence — the
block-table indirection [continuous batching](continuous-batching.md)'s
scheduler relies on to keep slots full. Reading that paged cache during
decode is itself a [memory coalescing](memory-coalescing.md) problem:
gathering one head's key vector across scattered blocks is the same
strided-access pattern that page counts in 128-byte transactions.

## Bytes per token, measured against model shape

Five published model shapes run through the same closed-form byte count —
`2 x layers x kv_heads x head_dim x itemsize(dtype)` per token — to see how
the GQA grouping ratio trades against depth.

| model | layers | heads | kv_heads | head_dim | dtype | bytes/token | KiB/token |
|---|---|---|---|---|---|---|---|
| Llama-2-7B | 32 | 32 | 32 | 128 | float16 | 524,288 | 512.0 |
| Llama-2-13B | 40 | 40 | 40 | 128 | float16 | 819,200 | 800.0 |
| Llama-2-70B | 80 | 64 | 8 | 128 | float16 | 327,680 | 320.0 |
| Llama-3-8B | 32 | 32 | 8 | 128 | float16 | 131,072 | 128.0 |
| Mistral-7B | 32 | 32 | 8 | 128 | float16 | 131,072 | 128.0 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def kv_cache_bytes_per_token(layers, kv_heads, head_dim, dtype):
    return 2 * layers * kv_heads * head_dim * np.dtype(dtype).itemsize

models = [
    ("Llama-2-7B",  32, 32, 32, 128, "float16"),
    ("Llama-2-13B", 40, 40, 40, 128, "float16"),
    ("Llama-2-70B", 80, 64, 8,  128, "float16"),
    ("Llama-3-8B",  32, 32, 8,  128, "float16"),
    ("Mistral-7B",  32, 32, 8,  128, "float16"),
]
for name, L, H, KVH, D, dt in models:
    b = kv_cache_bytes_per_token(L, KVH, D, dt)
    print(f"{name}: layers={L} heads={H} kv_heads={KVH} head_dim={D} dtype={dt} "
          f"bytes_per_token={b} kib_per_token={b/1024:.1f}")
PY
```

Llama-2-70B has 2.5x the layers of Llama-2-7B but a quarter of its KV
heads (8 against 32), and the two effects do not cancel: depth loses to
head count, landing 70B at 320 KiB/token against 7B's 512. Llama-3-8B and
Mistral-7B land on an identical 131,072 bytes/token despite different
hidden sizes and FFN widths, because neither appears in the formula —
cache cost is set only by `layers x kv_heads x head_dim`, so a bigger FFN
is close to free for the cache while more heads never is. The 4x gap
between Llama-2-7B's 32 kv_heads and Llama-3-8B's 8, at the same 32
layers, is the entire grouped-query-attention pitch stated in bytes.

## Cache versus weights: the context length at which one exceeds the other

Weight bytes come from the parameter-count approximation
$N \approx 12 \times \text{layers} \times d_{\text{model}}^2$ (Kaplan et
al. 2020; it ignores embeddings and biases, so treat it as within a few
percent of a real checkpoint), applied to two 32-layer, 4,096-wide, fp16
shapes that differ only in kv_heads.

| model shape | context (tokens) | cache bytes | fraction of total memory |
|---|---|---|---|
| MHA, 32 kv_heads | 2,048 | 1,073,741,824 | 7.69% |
| MHA, 32 kv_heads | 8,192 | 4,294,967,296 | 25.00% |
| MHA, 32 kv_heads | 24,576 | 12,884,901,888 | 50.00% |
| MHA, 32 kv_heads | 32,768 | 17,179,869,184 | 57.14% |
| GQA, 8 kv_heads | 24,576 | 3,221,225,472 | 20.00% |
| GQA, 8 kv_heads | 98,304 | 12,884,901,888 | 50.00% |
| GQA, 8 kv_heads | 131,072 | 17,179,869,184 | 57.14% |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

def kv_cache_bytes_per_token(layers, kv_heads, head_dim, dtype):
    return 2 * layers * kv_heads * head_dim * np.dtype(dtype).itemsize

def weight_bytes(layers, d_model, dtype):
    params = 12 * layers * d_model ** 2      # Kaplan et al. 2020, eq 2.1
    return params * np.dtype(dtype).itemsize

shapes = [
    ("MHA, 32 kv_heads", 32, 4096, 32, 128, "float16"),
    ("GQA, 8 kv_heads",  32, 4096, 8,  128, "float16"),
]
ctx_lengths = {
    "MHA, 32 kv_heads": [2048, 8192, 24576, 32768],
    "GQA, 8 kv_heads":  [24576, 98304, 131072],
}
for name, L, D_MODEL, KVH, HD, dt in shapes:
    wbytes = weight_bytes(L, D_MODEL, dt)
    bpt = kv_cache_bytes_per_token(L, KVH, HD, dt)
    crossover = wbytes // bpt
    print(f"{name}: weight_bytes={wbytes} ({wbytes/1024**3:.1f} GiB) "
          f"bytes_per_token={bpt} crossover_tokens={crossover}")
    for ctx in ctx_lengths[name]:
        cache_bytes = bpt * ctx
        frac = cache_bytes / (wbytes + cache_bytes)
        print(f"   ctx={ctx} cache_bytes={cache_bytes} fraction={frac*100:.2f}%")
PY
```

Both shapes carry the identical 12,884,901,888 bytes (12.0 GiB) of
weights, since the approximation never sees kv_heads — only the cache
side changes. That is why the crossover moves: the MHA shape's cache
reaches parity with its own weights at 24,576 tokens, the GQA shape not
until 98,304, precisely 4x further out, matching the 4x gap in
bytes/token above. Past crossover, memory is dominated by *how long the
conversation ran*, not by checkpoint size — the opposite of "load the
weights, that's the size."

## What recomputing instead of caching costs, in attention FLOPs

The same closed-form attention-FLOP formula the
[FLOP-count exercise](../tasks/rwb-attention-flop-count/task.md) grades —
`4 x batch x heads x seqlen_q x seqlen_k x head_dim`, halved for causal —
summed over generating $T$ tokens two ways: recompute-from-scratch every
step (`seqlen_q = seqlen_k = t`, causal) against incremental decode with a
cache (`seqlen_q = 1`, `seqlen_k = t`), for one Llama-2-7B-shaped request
(32 heads, head_dim 128, batch 1).

| T (tokens generated) | recompute FLOPs (no cache) | cached FLOPs | ratio |
|---|---|---|---|
| 128 | 5,793,906,688 | 135,266,304 | 42.8 |
| 512 | 367,578,316,800 | 2,151,677,952 | 170.8 |
| 2,048 | 23,473,430,724,608 | 34,376,515,584 | 682.8 |
| 8,192 | 1,501,474,764,881,920 | 549,822,922,752 | 2,730.8 |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
def attention_flops(batch, heads, seqlen_q, seqlen_k, head_dim, causal):
    total = 4 * batch * heads * seqlen_q * seqlen_k * head_dim
    if causal:
        total //= 2
    return total

def recompute_total(T, batch, heads, head_dim):
    return sum(attention_flops(batch, heads, t, t, head_dim, True) for t in range(1, T + 1))

def cached_total(T, batch, heads, head_dim):
    return sum(attention_flops(batch, heads, 1, t, head_dim, False) for t in range(1, T + 1))

BATCH, HEADS, HEAD_DIM = 1, 32, 128
for T in (128, 512, 2048, 8192):
    r = recompute_total(T, BATCH, HEADS, HEAD_DIM)
    c = cached_total(T, BATCH, HEADS, HEAD_DIM)
    print(f"T={T} recompute_flops={r} cached_flops={c} ratio={r/c:.1f}")
PY
```

The ratio does not settle at a constant — it grows with $T$, close to
$T/3$ in every row (42.8 at $T{=}128$, 2,730.8 at $T{=}8{,}192$, both
within 0.2% of $T/3$), because recompute sums a term quadratic in $t$
over $T$ steps ($O(T^3)$) while the cached path sums a term linear in
$t$ ($O(T^2)$); their ratio is $O(T)$. This table counts only the two
attention matmuls, not the $K$/$V$/$Q$ projections a genuinely cache-free
run would also redo for every past token every step, so the real gap is
larger still, not smaller.

## Practise it

```bash
mlsys grade llm-kv-cache-byte-model-layers-heads-d-ctx-dtype
```

[That task](../tasks/llm-kv-cache-byte-model-layers-heads-d-ctx-dtype/task.md)
gates `kv_cache_bytes(layers, heads, head_dim, seq_len, dtype)` on
`size_ratio == 1.0` against four shapes spanning float32, float16, int8,
and float64. The starter is `raise NotImplementedError`, an immediate
fail. The sharper failure: a NumPy-reduction implementation can return an
exactly correct `np.int64`, and the grader's `isinstance(got, int)` check
rejects it anyway — right value, wrong type, still zero.

More of the same accounting, in increasing scope:
[MHA vs GQA vs MQA in one function](../tasks/llm-kv-cache-bytes-mha-vs-gqa-vs-mqa/task.md),
[the same formula gated on tolerance instead of a boolean match](../tasks/sys-kv-cache-byte-size-formula/task.md),
[bytes for N contiguous request slots](../tasks/rwb-kv-memory-for-n-contiguous-slots/task.md),
[the byte-reduction ratio a grouping choice buys](../tasks/rwa-measure-the-kv-cache-byte-reduction-ratio/task.md),
and [bytes actually moved per decode step](../tasks/llm-kv-bytes-moved-per-decode-step/task.md)
— read traffic, not stored size. For the FLOP side:
[attention FLOP counting](../tasks/rwb-attention-flop-count/task.md).
For the layout a decode-step read contends with:
[coalesced KV-cache read layout design](../tasks/gpu-coalesced-kv-cache-read-layout-design/task.md)
and [prefetching KV-cache reads for decode](../tasks/cpu-prefetch-kv-cache-reads-for-decode/task.md).

## Common mistakes

- **Confusing KV heads with query heads.** GQA/MQA shrink `kv_heads`
  only; Llama-3-8B's 8 kv_heads cost 131,072 bytes/token, a quarter of
  Llama-2-7B's 524,288 at the same 32 layers.
- **Assuming weights dominate memory at any practical context.** A
  32-layer, 4,096-wide MHA model's cache equals its 12.0 GiB of weights
  by 24,576 tokens — inside a single long conversation.
- **Not re-deriving bytes-per-token after changing dtype.** `int8`,
  `float16`, `float32` carry itemsizes 1, 2, 4; a dtype swap without
  recomputing `bytes_per_token` silently shifts memory-fit math 2x or 4x.
- **Returning the right number as the wrong type.** A NumPy-reduction
  byte-count can be numerically exact and still fail `isinstance(got,
  int)`, because it comes back as `np.int64`, not a plain `int`.
- **Benchmarking recompute-vs-cache at one short $T$.** The FLOP ratio
  is close to $T/3$, so $T{=}128$ (42.8x) understates caching's value by
  roughly 64x against $T{=}8{,}192$ (2,730.8x).

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md),
which lists attention and KV-cache as **124 tasks in this bank, some
overlap** — cache management specifically has no graded practice
elsewhere:

- **[LeetGPU — challenge set](https://leetgpu.com/challenges)** — ~90
  browser-graded kernel challenges, one `int8-kv-cache-attention`
  exercise; free tier is a CPU-emulated GPU, no block-table coverage.
- **[tspeterkim/paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal)**
  — smallest readable block-table manager found, on a real Llama-3
  forward pass; a reference to read, not to fail against.
- **[jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)** — reference
  implementation of 2-bit asymmetric KV-cache quantization, the numerics
  behind tasks like `rwq-per-token-int8-kv-quant-attention-error`.
- **[vLLM — PagedAttention design doc](https://docs.vllm.ai/en/latest/design/paged_attention/)**
  — primary reference for block-table, paged-KV layout in production;
  a walkthrough plus source, not something graded.

## References

1. Pope, R. et al., *Efficiently Scaling Transformer Inference*, MLSys 2023
   — formalizes KV-cache memory and the compute/memory tradeoffs of decode.
   https://arxiv.org/abs/2211.05102
2. Kaplan, J. et al., *Scaling Laws for Neural Language Models*, 2020 —
   source of the $N \approx 12 \, n_{\text{layer}} \, d_{\text{model}}^2$
   parameter approximation used in the weights-vs-cache table.
   https://arxiv.org/abs/2001.08361
3. Ainslie, J. et al., *GQA: Training Generalized Multi-Query Transformer
   Models from Multi-Head Checkpoints*, EMNLP 2023 — the grouped-query
   mechanism behind the kv_heads column. https://arxiv.org/abs/2305.13245
4. Kwon, W. et al., *Efficient Memory Management for Large Language Model
   Serving with PagedAttention*, SOSP 2023.
   https://arxiv.org/abs/2309.06180
