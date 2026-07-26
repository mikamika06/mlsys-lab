---
title: "What is llm inference?"
description: "LLM inference explained, with a measured arithmetic-intensity table for prefill vs. decode across model size and batch that you can reproduce, plus a graded roofline exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is llm inference?

LLM inference is the forward-only run of a trained transformer that turns a
prompt into generated tokens, split into two phases with opposite hardware
behavior: prefill, which scores the whole prompt in one batched pass, and
decode, which emits one token per step. At a 4,096-token context, decode's
arithmetic intensity is pinned near 0.5 FLOPs/byte no matter how large the
model gets, while prefill sits above 1,700 FLOPs/byte for every model size
below. The table further down counts exactly where that gap comes from, and
how far batching can close it on the decode side.

## How it works

Every request passes through two phases that share weights but not access
patterns. **Prefill** takes the whole prompt — a `(P, d_model)` block of
tokens — through every layer in one batched forward pass: each weight
matrix is read from memory once and reused across all `P` tokens' worth of
matmuls. **Decode** then runs autoregressively, one new token at a time,
attending to every key and value already sitting in the KV cache, with the
very same weight matrices loaded again to produce that single token. The
cache grows by one row per step and is read back in full on the next one —
the same "one fixed resource, many contenders" shape that
[continuous batching](continuous-batching.md) manages at the scheduler
level, deciding which waiting request gets the next free decode slot once
the phase split below has told you which phase is worth scheduling around.

Inside a layer, both phases run the same operations — project to `Q, K, V`,
score and weight the attention, project back through `W_O`, then the FFN —
but two things around them differ from training. There is no backward
pass, so nothing here needs [gradient checkpointing](gradient-checkpointing.md):
that technique trades recomputation for saved activations across a backward
pass that inference never runs. Numerics differ too: served weights are
almost always [bfloat16 or float16](bfloat16-vs-float16.md) rather than
float32, loaded from whichever [gguf or safetensors](gguf-vs-safetensors.md)
container the engine reads, and decompressed to that working dtype before
the first matmul. Inside attention, the score row becomes weights through a
[softmax, not a sigmoid](softmax-vs-sigmoid.md), because the row has to sum
to one across every cached position it attends to; the normalization
between sub-blocks is usually [RMSNorm rather than LayerNorm](rmsnorm-vs-layernorm.md),
dropping the mean-subtraction term that page counts the cost of.

None of this is timed below. What gets counted is FLOPs and bytes moved per
token, because that ratio — arithmetic intensity — decides, before a single
kernel runs, whether a phase is limited by the GPU's arithmetic units or by
how fast it can pull weights and cache out of HBM. The GPU-side version of
the same "bytes moved decides everything" idea, at a much finer grain, is
[memory coalescing](memory-coalescing.md): there the countable unit is a
128-byte transaction; here it is a whole weight matrix's worth of bytes,
amortized across however many tokens share that one read.

## Arithmetic intensity of prefill vs. decode, across model size and batch

One transformer layer, fp16 weights, a 4,096-token prompt for prefill and a
4,096-token cache for decode. `decode_ai` extends the batch-1 formula from
[Prefill vs Decode: Roofline Placement](../tasks/sys-prefill-vs-decode-roofline-placement/task.md)
to batch `B`: `B` independent requests each with 4,096 cached tokens, all
decoding their next token in the same step and sharing one weight load —
which is exactly what continuous batching makes possible. The ridge point
uses an A100 80GB SXM's published specs: 312 TFLOP/s dense fp16 tensor-core
throughput over 2,039 GB/s of HBM2e bandwidth.

| hidden size H | prefill AI (P=4,096) | decode AI, batch 1 | decode AI, batch 8 | decode AI, batch 64 |
|---|---|---|---|---|
| 1,024 | 1,706.67 | 0.500 | 1.052 | 1.221 |
| 4,096 | 1,911.47 | 0.500 | 2.000 | 3.199 |
| 8,192 | 1,972.15 | 0.500 | 2.600 | **5.472** |

Reproduce it — pure arithmetic on closed-form FLOP and byte counts, so the
numbers are identical on every machine:

```bash
pip install mlsys-lab
python3 - <<'PY'
D = 2                 # fp16, bytes per element
P = S = 4096          # prefill length == decode's cached context length
PEAK_FLOPS = 312e12   # A100 80GB SXM, dense fp16 tensor-core peak
PEAK_BYTES = 2039e9   # A100 80GB SXM, published HBM2e bandwidth
RIDGE = PEAK_FLOPS / PEAK_BYTES

def weight_bytes(H):
    return 12 * H**2 * D                      # QKV + O + FFN(H->4H->H)

def prefill_ai(H, P):
    flops = 12 * H**2 * P + 2 * H * P**2
    bts = weight_bytes(H) + 3 * D * P * H
    return flops / bts

def decode_ai(H, S, B):
    flops = B * (12 * H**2 + 2 * H * S)
    bts = weight_bytes(H) + D * B * (2 * S * H + 3 * H)
    return flops / bts

print(f"ridge={RIDGE:.2f}")
for H in (1024, 4096, 8192):
    pai = prefill_ai(H, P)
    d1, d8, d64 = (decode_ai(H, S, B) for B in (1, 8, 64))
    print(f"H={H:>5}  prefill_ai={pai:.2f}  decode_ai_b1={d1:.3f}  "
          f"decode_ai_b8={d8:.3f}  decode_ai_b64={d64:.3f}")

lo, hi, H = 1.0, 1000.0, 4096
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if prefill_ai(H, mid) < RIDGE else (lo, mid)
print(f"crossover_P_H4096={round(hi)}")
PY
```

Prefill clears the ridge point of 153.02 FLOPs/byte at every size tested —
compute-bound — while decode at batch 1 sits at exactly 0.500 FLOPs/byte
regardless of `H`, because both its FLOPs and its dominant byte cost scale
as `H²` and cancel: this repo's own
[batch-1 CUDA version of this same placement](../tasks/gpu-roofline-of-decode-vs-prefill-batch-1/task.md)
calls that same constant a "0.5 floor". Batching decode narrows the gap —
64 concurrent requests push `H=8,192`'s decode AI to 5.472 — but even there
it stays roughly 28x below the ridge, so decode remains memory-bound at any
batch size a single accelerator can hold in KV cache. The two curves only
meet at the far end no serving system runs at: prefill's own AI does not
clear the ridge until the prompt exceeds 308 tokens either, for `H=4,096`.

## Practise it

```bash
mlsys grade sys-prefill-vs-decode-roofline-placement
```

[That task](../tasks/sys-prefill-vs-decode-roofline-placement/task.md) gates
on `modeled_arith_intensity <= 1e-6` across all six returned FLOPs/bytes/AI
fields for both phases, plus `classification_exact == 1.0` on which side of
the ridge point each phase lands. The shipped starter is
`raise NotImplementedError`, so it fails immediately; the instructive way to
fail it once you've written something is dropping the `12H²·dtype_bytes`
weight-read term from decode's byte count and keeping only the KV-cache
read — that inflates decode's AI well past the ridge and misclassifies a
bandwidth-bound step as compute-bound.

In roughly increasing difficulty:
[the matmul FLOP ratio alone, graded against a real NumPy forward pass rather than a formula](../tasks/llm-prefill-vs-decode-flop-ratio/task.md),
[prefill's closed-form arithmetic intensity in isolation](../tasks/llm-prefill-arithmetic-intensity-closed-form/task.md),
[decode's closed-form arithmetic intensity in isolation](../tasks/llm-decode-arithmetic-intensity-closed-form/task.md),
[the bytes moved by one decode step's KV-cache read alone](../tasks/llm-kv-bytes-moved-per-decode-step/task.md),
[the same placement problem worked from a CUDA kernel instead of Python](../tasks/gpu-roofline-of-decode-vs-prefill-batch-1/task.md),
[a simpler CPU-side classify-only version](../tasks/cpu-prefill-vs-decode-roofline-placement/task.md),
and [classifying compute- vs memory-bound regimes across many configurations at once, vectorized in NumPy](../tasks/llm-predict-compute-vs-memory-bound-regime/task.md).

## Common mistakes

- **Sizing serving hardware off prefill's FLOP number.** Prefill's FLOPs
  dwarf a single decode step's, but a serving fleet spends nearly all its
  wall-clock time decoding — provisioning for prefill's compute and
  ignoring decode's 0.500 FLOPs/byte floor under-provisions memory
  bandwidth, the resource decode actually contends for.
- **Assuming any prompt makes prefill compute-bound.** At `H=4,096` prefill
  only clears the 153.02 ridge point past 308 tokens; a short system
  message or single-sentence prompt is itself bandwidth-bound, same as
  decode, because there aren't enough tokens yet to amortize the one
  weight read.
- **Treating decode batching as free.** Batching pushes `H=8,192`'s decode
  AI from 0.500 to 5.472 at batch 64 — real progress — but every one of
  those 64 requests still needs its own KV cache resident in memory, so the
  win is bounded by however many caches actually fit, not by arithmetic
  alone.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md):

- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** — a
  from-scratch vLLM reimplementation in about 1,200 readable lines, with a
  real scheduler and block manager that actually split prefill and decode
  work the way this page describes. No test suite; you read the code and
  run `bench.py` rather than get a scored answer.
- **[tiny-vllm](https://github.com/jmaczan/tiny-vllm)** — a guided
  "build vLLM yourself" repo in C++/CUDA that works explicitly through
  prefill vs. decode, static and continuous batching, and PagedAttention.
  No automated grading; you compare against the author's reference by eye.
- **[achi9629/llm-inference-engine](https://github.com/achi9629/llm-inference-engine)**
  — a solo project building an inference engine in explicit stages (plain
  forward pass, KV cache, static batching, continuous batching, paged KV
  cache) with 122 pytest tests per stage. The closest GitHub match to
  "implement the mechanic, then check yourself" for this topic, but a
  young, one-star, single-author repo.
- **[Vidur](https://github.com/microsoft/vidur)** — Microsoft Research's
  LLM-serving simulator (MLSys 2024): configure a model, hardware and
  scheduling policy, replay a trace, and read back TTFT/TPOT numbers. A
  genuine research tool for scheduling tradeoffs, with no notion of a
  correct answer to grade against.
- **Efficiently Serving LLMs (DeepLearning.AI / Predibase)** — a 2h40m
  video course covering KV caching, batching and continuous batching with
  seven run-along notebooks. Free tier is watch-and-run-the-cell; the one
  graded assignment sits behind a paid tier.

## References

1. Williams, S., Waterman, A., Patterson, D., *Roofline: An Insightful
   Visual Performance Model for Multicore Architectures*, Communications
   of the ACM, 2009 — the roofline model this page's ridge point comes
   from. https://dl.acm.org/doi/10.1145/1498765.1498785
2. NVIDIA, *NVIDIA A100 Tensor Core GPU Architecture* whitepaper — source
   of the 312 TFLOP/s fp16 and 2,039 GB/s HBM2e specs used for the ridge
   point above. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
3. Kwon, W. et al., *Efficient Memory Management for Large Language Model
   Serving with PagedAttention*, SOSP 2023 — the KV-cache accounting this
   page's decode byte count follows. https://arxiv.org/abs/2309.06180
