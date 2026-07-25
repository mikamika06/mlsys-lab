## Context

Tensor cores compute matmuls in **mixed precision**: both input operands
are rounded to fp16 before every multiply, but the running sum is kept in
a full-precision (fp32) accumulator. That split is deliberate — fp16
inputs halve memory bandwidth and let the hardware pack more multiplies
per cycle, while the fp32 accumulator prevents the *sum* (which can have
hundreds of terms, for a real attention head) from compounding rounding
error on top of rounding error.

This task computes attention scores, $S = QK^\top / \sqrt{D}$, the same
way: round each element of $Q$ and $K$ to fp16 precision, multiply the
rounded values, and accumulate the products in full precision. $Q$ and
$K$ are restricted to $[1, 2)$ here specifically because fp16's 10
explicit mantissa bits give an *exact*, analyzable ULP of $1/1024$ across
that whole range — `floorf(x*1024 + 0.5) / 1024` is genuine fp16
rounding there, not an approximation.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void qkt_mixed_precision(float* S, const float* Q, const float* K, int M, int N, int D);
```

For `idx = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `idx < M*N`:
`i = idx/N`, `j = idx%N`. Accumulate, for `d` in `[0,D)`:
`fp16(Q[i*D+d]) * fp16(K[j*D+d])`, where
`fp16(x) = floorf(x*1024.0f + 0.5f) / 1024.0f`. Write
`S[idx] = acc / sqrtf(D)`.

## Example

`Q[0] = 1.30012, K[0] = 1.70004` (both in `[1,2)`): fp16-round each first —
`fp16(1.30012) = floor(1331.32+0.5)/1024 = 1331/1024 ≈ 1.29980`,
`fp16(1.70004) = floor(1740.84+0.5)/1024 ≈ 1.70020` — then multiply the
*rounded* values, not the originals. Rounding the inputs before
multiplying (rather than rounding the product, or not rounding at all)
introduces a small, real, per-term error — on this task's fixed `4x4`,
`D=8` fixture, the fully-rounded score matrix differs from the
unrounded-fp32 one by up to `~0.0011`, purely from that input rounding.

## What the gate checks

`max_abs_err <= 1e-6` against a numpy oracle that fp16-rounds `Q` and `K`
the identical way before multiplying, on a fixed `M=N=4, D=8` fixture.
Rounding the accumulator instead of the inputs, rounding the product
instead of each factor separately, or skipping the rounding step
entirely, all diverge from the reference score matrix by more than the
tolerance.
