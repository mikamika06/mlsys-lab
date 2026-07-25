## Context

The roofline model classifies an operation by its arithmetic intensity —
FLOPs performed per byte of memory traffic — against the hardware's
**ridge point**, $\text{peak\_flops} / \text{peak\_bandwidth}$: below the
ridge, the op can't keep the compute units fed fast enough and is
**memory-bound**; at or above it, memory can supply data faster than the
compute units can consume it, and the op is **compute-bound**.

Four canonical LLM ops, with fixed FLOP/byte-traffic models (bytes
assume `float32`, everything read/written exactly once — no reuse, no
tiling):

- **GEMM** ($M \times K \times N$): $\text{flops} = 2MKN$,
  $\text{bytes} = 4(MK + KN + MN)$.
- **Attention** (sequence length $S$, head dim $D$): $\text{flops} =
  4S^2D$ ($QK^\top$ and $\text{softmax} \cdot V$ are each $2S^2D$),
  $\text{bytes} = 4(3SD + S^2 + SD)$ ($Q,K,V$ read once each, one $S
  \times S$ score matrix, one $S \times D$ output).
- **LayerNorm** ($N$ elements): $\text{flops} = 5N$ (mean, variance,
  normalize), $\text{bytes} = 4 \cdot 2N$ (read + write).
- **Elementwise** ($N$ elements): $\text{flops} = N$, $\text{bytes} = 4
  \cdot 2N$.

GEMM and attention pack many FLOPs per byte moved (every loaded value
gets reused across many multiply-adds); LayerNorm and elementwise touch
each byte for only a handful of FLOPs each — the classic reason attention
and GEMM scale onto compute-bound hardware while normalization and
activation layers stay memory-bound almost regardless of size.

## Task

Write a CUDA-C kernel (single thread):

```cpp
__global__ void roofline_rank(float* out,
                                float gemm_m, float gemm_k, float gemm_n,
                                float attn_s, float attn_d,
                                float ln_n, float ew_n,
                                float peak_flops, float peak_bw);
```

Compute `ridge = peak_flops / peak_bw`, then for GEMM, attention,
LayerNorm, and elementwise (in that order) using the formulas above:
`out[0..3]` = each op's arithmetic intensity, `out[4..7]` = `1.0` if that
op's AI $\ge$ ridge (compute-bound) else `0.0` (memory-bound).

## Example

With `gemm_m=gemm_k=gemm_n=64`, `attn_s=attn_d=64`, `ln_n=ew_n=4096`,
`peak_flops=1000`, `peak_bw=100` (ridge $= 10.0$):

| op | AI | bound |
|---|---|---|
| GEMM | $10.6\overline{6}$ | compute (`1.0`) |
| attention | $12.8$ | compute (`1.0`) |
| LayerNorm | $0.625$ | memory (`0.0`) |
| elementwise | $0.125$ | memory (`0.0`) |

GEMM and attention clear the ridge point comfortably; LayerNorm and
elementwise sit more than an order of magnitude below it — attention
even edges out GEMM here, both firmly compute-bound while the other two
are firmly memory-bound.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU over the fixed sizes above, requiring
`max_abs_err <= 1e-6` against all 8 values (4 AI numbers, 4 bound labels)
computed directly in Python from the same formulas. Swapping which
op-pair is compute- vs memory-bound, or using `>` instead of `>=` at the
exact ridge point, produces at least one label that doesn't match and
fails the gate.
