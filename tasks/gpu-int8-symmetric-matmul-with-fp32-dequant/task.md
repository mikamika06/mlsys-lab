## Context

Weight-only int8 quantization keeps activations at full precision but
stores weights as 8-bit integers plus a single **scale**: a real
number `s` such that `W ≈ round(W/s) * s`, with `round(W/s)` clamped to
the symmetric int8 range `[-127, 127]` (zero-point `0`, since the
values are centered around zero). This shrinks weight storage 4x
versus `fp32` at the cost of a small, bounded rounding error per
element.

A **dequant-fused** kernel never materializes the dequantized weight
matrix separately — it quantizes and immediately dequantizes each
weight value right where it's used, inline in the matmul's inner loop,
so the extra int8 storage saved never has to be paid back as extra
memory traffic for a full-precision copy.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void int8_dequant_matmul(const float* A, const float* W, float* C,
                                     int M, int N, int K, float scale);
```

One thread per output element (`idx = blockIdx.x*blockDim.x +
threadIdx.x`, `i = idx/N`, `j = idx%N`). For that `(i,j)`, compute

$$
C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot \mathrm{dequant}(W_{k,j})
$$

where `dequant(w)` rounds `w` to the nearest integer multiple of
`scale`, clamped to the int8 range: `q = round(|w|/scale)`, clamped to
`127` if larger, then `dequant(w) = sign(w) * q * scale`. `A` is used
at full precision, untouched.

## Example

`w = 0.32`, `scale = 0.01`: `q = round(0.32/0.01) = round(32) = 32`
(within `[0,127]`, no clamping needed), `dequant(w) = 32 * 0.01 =
0.32` — exact here since `0.32` happened to be a clean multiple of the
scale. `w = 0.317`: `q = round(31.7) = 32`, `dequant(w) = 0.32` — off
by `0.003`, the quantization error.

## What the gate checks

The grader launches `int8_dequant_matmul` on a fixed `8x8` output over
a depth-16 contraction (`A` in `[-2,2]`, `W` in `[-1,1]`), with `scale`
computed on the host as `max(|W|)/127` (a real per-tensor symmetric
scale), and compares the result against an exact `A @ W` (numpy). It
requires

$$
5\times10^{-4} \le \mathrm{rel\_err} \le 2\times10^{-2}
$$

The lower bound matters: a kernel that ignores `scale` and just
computes an ordinary full-precision matmul would trivially satisfy the
upper bound (skipping quantization can only make the result *more*
accurate) but reports `rel_err` around `10^{-16}` — caught by the floor.
On this fixture, correctly quantizing and dequantizing measures
`rel_err = 0.00495`: real, bounded, and consistent with what an int8
weight format should cost you.
