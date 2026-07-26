---
title: "What is LLM inference optimization?"
description: "LLM inference optimization explained as four independent levers, with a measured table of what each one buys on one fixed model shape, plus graded exercises."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is LLM inference optimization?

LLM inference optimization is the practice of cutting the cost of autoregressive decoding by
attacking whichever resource limits it — cache bytes, memory bandwidth, or allocator waste —
rather than adding compute the step was never short of. Four levers do almost all
of the work: KV-cache compression, cache quantization, batching, and paged allocation. One fixed
model shape below makes their sizes directly comparable.

## How it works

A single decode step reads one new token's activations through every layer and produces one new
token, so its useful arithmetic is tiny. What it cannot skip is reading the whole model's weights
and the whole KV cache accumulated so far — a step that is almost always memory-bound, not
compute-bound. Every lever below works by shrinking one of those two numbers, or by amortizing
them over more useful work, and the levers are independent enough to stack.

### KV cache compression

The KV cache stores one key and one value vector per layer, per attended head, per past token, so
its size is `2 · layers · heads · head_dim · context_length · bytes_per_element` — linear in every
factor. **Grouped-query attention (GQA)** cuts the `heads` factor directly: instead of one KV head
per query head (multi-head attention, MHA), several query heads share one KV head, and the cache
shrinks by exactly the group factor regardless of model size, layer count, or context length —
[the KV-cache byte formula](../tasks/sys-kv-cache-byte-size-formula/task.md) makes that ratio
explicit. Multi-query attention (MQA) is the extreme case, one KV head total, and
[MQA is just GQA at group size equal to the head count](../tasks/llm-mqa-is-the-gqa-limit-n-kv-1/task.md).
[KV-cache bytes for MHA, GQA, and MQA at once](../tasks/llm-kv-cache-bytes-mha-vs-gqa-vs-mqa/task.md)
is the task that computes all three from one config.

**Cache quantization** shrinks the `bytes_per_element` factor instead of the `heads` factor, and
is fully independent of it: any GQA or MQA cache can additionally be stored in
[int8](../tasks/rwq-per-token-int8-kv-quant-attention-error/task.md) or
[fp8](../tasks/rwc-per-tensor-fp8-e4m3-kv-quant-attention/task.md) rather than fp16, at the same
symmetric-range cost measured for
[weights and activations generally](integer-quantization-ranges.md) and for
[GGUF](gguf-vs-safetensors.md)'s block-quantized formats. The two levers multiply different
factors in the same formula, so they compound: a model already running GQA still gets the full
quantization discount on top.

### Batching

Batching does not touch the cache at all — it changes what the memory traffic is *for*. Model
weights are read once per batched matmul no matter how many sequences share it, so growing the
batch size amortizes that fixed weight-read cost over more useful FLOPs, while only the
per-sequence KV-cache reads grow with the batch.
[The closed-form decode arithmetic-intensity formula](../tasks/llm-decode-arithmetic-intensity-closed-form/task.md)
separates the weight term from the KV-cache term, which is what makes the batching effect
computable rather than asserted; the same shift plays out per head under
[GQA specifically](../tasks/llm-gqa-effect-on-decode-arithmetic-intensity/task.md), and which batch
size clears a given hardware's compute/bandwidth ratio is
[its own task](../tasks/llm-operational-intensity-crossover-batch-size/task.md). It is the same
amortization idea behind [continuous batching](continuous-batching.md) keeping slots full — a full
batch pays off only if the scheduler can assemble one.

### Paged allocation

Even a compressed cache still has to live somewhere, and the naive layout —
reserve `max_context_length` slots per sequence up front — wastes the entire unused tail of every
sequence shorter than the cap. A paged allocator instead hands out fixed-size blocks on demand, so
the only waste is the partially-filled last block per sequence.
[Internal fragmentation, paged against contiguous pre-allocation](../tasks/rwb-internal-fragmentation-paged-vs-contiguous-pre-alloc/task.md)
computes exactly that gap from a batch of sequence lengths, and
[a paged block allocator run over a real arrival trace](../tasks/rwb-paged-block-allocator-over-an-arrival-free-trace/task.md)
does it under actual admission and eviction. Reading a paged cache back means indirecting through
a block table rather than one flat offset —
[block-table gather](../tasks/llm-pagedattention-block-table-gather/task.md) — and once the blocks
are gathered, laying them out so the read is
[coalesced](memory-coalescing.md) rather than scattered is
[its own layout problem](../tasks/gpu-coalesced-kv-cache-read-layout-design/task.md).

## The four levers on one fixed model shape

Same model throughout — 32 layers, 4,096 hidden size, 32 query heads, 128-dim heads, a 4,096-token
context — so the columns are comparable. GQA and quantization act on cache bytes; batching acts on
arithmetic intensity (FLOPs per byte moved); paging acts on wasted allocator bytes for a batch of
64 sequences (lengths drawn from `numpy.random.default_rng(0)`, block size 16, 4,096-token cap).

| lever | before | after | change | measures |
|---|---|---|---|---|
| MHA → GQA (32 heads → 8 KV heads) | 2.0 GiB | 0.5 GiB | ÷4 | KV-cache bytes |
| fp16 → int8 cache (same GQA config) | 0.5 GiB | 0.25 GiB | ÷2 | KV-cache bytes |
| batch 1 → batch 64 (same shape) | 0.29 FLOPs/byte | 13.99 FLOPs/byte | ×48.2 | decode arithmetic intensity |
| contiguous → paged allocation (64 seqs) | 8,031 MiB wasted | 30 MiB wasted | −8,001 MiB | allocator waste |

Reproduce it:

```bash
pip install mlsys-lab
python3 - <<'PY'
import numpy as np

L, H, n_q, d_h, n_kv, S = 32, 4096, 32, 128, 8, 4096
FP16, INT8 = 2, 1
GiB, MiB, KiB = 2**30, 2**20, 2**10

def kv_bytes(n, dtype_bytes):
    return 2 * L * n * d_h * S * dtype_bytes

kv_mha = kv_bytes(n_q, FP16)
kv_gqa = kv_bytes(n_kv, FP16)
kv_gqa_int8 = kv_bytes(n_kv, INT8)
print("kv_mha_GiB", kv_mha / GiB)
print("kv_gqa_GiB", kv_gqa / GiB)
print("kv_gqa_int8_GiB", kv_gqa_int8 / GiB)
print("gqa_ratio", kv_gqa / kv_mha)
print("int8_ratio", kv_gqa_int8 / kv_gqa)

N_h = n_q
def ops(H, S): return 12 * H**2 + 2 * H * S
def bytes_(H, S, N_h, B): return 48 * H**2 + B * (8 * S * H / N_h + 4 * H)
def ai(H, S, N_h, B): return (B * ops(H, S)) / bytes_(H, S, N_h, B)

ai1 = ai(H, S, N_h, 1)
ai64 = ai(H, S, N_h, 64)
print("ai1", round(ai1, 3))
print("ai64", round(ai64, 3))
print("ai_ratio", round(ai64 / ai1, 1))

per_token_bytes = kv_gqa_int8 / S
print("per_token_KiB", per_token_bytes / KiB)

rng = np.random.default_rng(0)
lengths = rng.integers(1, S, size=64)

def internal_fragmentation(seqlens, block_size, max_len):
    seqlens = np.asarray(seqlens)
    paged = np.ceil(seqlens / block_size) * block_size - seqlens
    contig = max_len - seqlens
    return paged.sum(), contig.sum()

paged_waste_tok, contig_waste_tok = internal_fragmentation(lengths, 16, S)
print("paged_waste_tok", int(paged_waste_tok))
print("contig_waste_tok", int(contig_waste_tok))

paged_MiB = paged_waste_tok * per_token_bytes / MiB
contig_MiB = contig_waste_tok * per_token_bytes / MiB
saved_MiB = contig_MiB - paged_MiB
print("paged_MiB", paged_MiB)
print("contig_MiB", contig_MiB)
print("saved_MiB", saved_MiB)
print("saved_ratio", round(contig_MiB / paged_MiB, 1))
PY
```

The four columns are not on the same axis, and reading them as if they were is the mistake this
table exists to prevent. GQA and int8 both shrink the *same* quantity — cache bytes — so they
compound: 8 KV heads at int8 is a flat quarter times a flat half, 0.25 GiB against a 2.0 GiB
MHA-fp16 baseline, an 8x reduction from two independent, stackable decisions. Batching's 48.2x
sits on a different axis — arithmetic intensity, not footprint — because the fixed 0.5 GiB of
weight bytes amortizes over 64x the FLOPs while the KV-cache-read term barely grows; it buys
nothing back on disk. Paging's saving is workload-shaped: 128,496 wasted token-slots under
contiguous pre-allocation fall to 480 once only each sequence's last block goes unfilled, a
267.7x gap that tracks the batch's length variance the same way
[continuous batching's win](continuous-batching.md) tracks output-length spread.

## Practise it

```bash
mlsys grade llm-kv-cache-bytes-mha-vs-gqa-vs-mqa
```

[That task](../tasks/llm-kv-cache-bytes-mha-vs-gqa-vs-mqa/task.md) gates two ratios —
`gqa_mha_ratio` and `mqa_mha_ratio` — against a NumPy oracle at `1e-9` relative tolerance, which
means getting the *absolute* byte count right is not enough on its own if the per-variant grouping
is wrong; a common failure is applying the group factor to `d_model` instead of to the head count,
which cancels out for MHA (`groups == heads`) and only shows up once the grader checks GQA and MQA
too.

The other three levers, roughly in the order they appear above: the KV-cache byte formula in
isolation,
[per-token int8 KV quantization error](../tasks/rwq-per-token-int8-kv-quant-attention-error/task.md),
[decode arithmetic intensity in closed form](../tasks/llm-decode-arithmetic-intensity-closed-form/task.md),
[the batch size that clears a hardware's compute/bandwidth crossover](../tasks/llm-operational-intensity-crossover-batch-size/task.md),
and internal fragmentation, paged vs. contiguous.

## Common mistakes

- **Treating the four levers as substitutes.** They are not competing for the same budget — a
  server can run GQA, quantize the cache to int8, batch aggressively, and page its allocator, all
  at once, and production engines do exactly that.
- **Quoting GQA's saving without saying against what.** "GQA cuts the cache 4x" is only true at a
  fixed group factor; [MQA](../tasks/llm-mqa-is-the-gqa-limit-n-kv-1/task.md) at the same model
  is a much larger cut, and a 2-group GQA config is a much smaller one.
- **Measuring batching's win in bytes instead of arithmetic intensity.** Batching barely changes
  how much memory a decode step touches — [it changes what that traffic buys](../tasks/llm-kv-bytes-moved-per-decode-step/task.md),
  which is a FLOPs-per-byte question, not a footprint question.
- **Assuming paged allocation always saves 267.7x.** That ratio is a property of *this* batch's
  length spread against a 4,096-token cap; a batch whose sequences all sit near the cap has almost
  nothing for paging to reclaim, the same way
  [fp8 vs. fp16 KV concurrency](../tasks/rwb-kv-memory-and-max-concurrency-fp8-vs-fp16/task.md)
  depends on the workload, not just the dtype.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[vLLM](https://github.com/vllm-project/vllm)** — the production system that popularized
  paged KV-cache allocation and continuous batching together; real, heavily used code with
  benchmarks, and no exercises to check your own understanding against.
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** — roughly 1,200 lines
  reimplementing the same scheduler and block manager readably; the best single file for seeing
  how the four levers interact in one codebase, still with no scored answer.
- **[Efficiently Serving LLMs (DeepLearning.AI / Predibase)](https://www.deeplearning.ai/short-courses/efficiently-serving-llms/)**
  — a short course covering KV caching, continuous batching, and quantized serving with
  run-along notebooks; a graded assignment exists only on the paid tier.
- **Hugging Face's own KV-cache and quantization guides**, reference 3 below, are the clearest
  prose explanation of why each lever exists; neither one computes a number for you to check.

## References

1. Kwon, W. et al., *Efficient Memory Management for Large Language Model Serving with
   PagedAttention*, SOSP 2023 — paged KV-cache allocation and its fragmentation savings.
   https://arxiv.org/abs/2309.06180
2. Ainslie, J. et al., *GQA: Training Generalized Multi-Query Transformer Models from
   Multi-Head Checkpoints*, EMNLP 2023 — the group-factor tradeoff this page's KV-cache rows
   measure. https://arxiv.org/abs/2305.13245
3. Hugging Face, *KV Cache Quantization* and *Best Practices for Generation with Cache*.
   https://huggingface.co/docs/transformers/main/en/kv_cache
