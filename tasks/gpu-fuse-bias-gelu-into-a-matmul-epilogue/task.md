## Context

A transformer FFN block computes `GELU(X @ W + bias)`. The naive way to
run this on a GPU is three separate passes over global memory: one
kernel computes `Z = X @ W`, writing the full result out; a second adds
`bias`; a third applies `GELU`. Every intermediate value gets written to
global memory and read straight back, even though nothing outside that
one element's own computation ever needed it.

**Kernel fusion** collapses all three into one: each thread computes its
own output element's dot product in a register, adds its bias term, and
applies the activation -- all before the result ever touches memory. The
raw `X @ W` value and the "bias-added, not yet activated" value never
exist anywhere but that one register.

The **GELU** activation used here is the standard tanh approximation:

$$
\mathrm{GELU}(z) = 0.5\,z\left(1 + \tanh\!\Big(\sqrt{2/\pi}\,(z + 0.044715\,z^3)\Big)\right)
$$

This CUDA-C subset has no `tanh` builtin, but `tanh` has an exact
closed form in terms of `exp` (which *is* available):

$$
\tanh(u) = \frac{e^{2u} - 1}{e^{2u} + 1}
$$

## Task

Implement:

```cuda
__global__ void matmul_bias_gelu(float* out, const float* A, const float* B, const float* bias, int M, int K, int N);
```

`A` is a flattened $M \times K$ matrix, `B` a flattened $K \times N$
matrix, `bias` has $N$ entries, `out` is $M \times N$ -- all row-major.
Launch with one block of `M*N` threads (this task fixes
`M=4, K=8, N=4`, so `16` threads), one per output element:

1. Thread `tid = threadIdx.x` owns `row = tid / N`, `col = tid % N`.
2. Reduce `acc = sum_k A[row*K+k] * B[k*N+col]` in a register.
3. `z = acc + bias[col]`.
4. `u = 0.7978845608 * (z + 0.044715*z*z*z)`; `t = (expf(2*u)-1) /
   (expf(2*u)+1)`; `g = 0.5*z*(1+t)`.
5. `out[row*N+col] = g` -- write only this final value; never write `z`
   (or any other intermediate) to global memory.

## Example

For a `1x1x1` toy case, `A=[2]`, `B=[3]`, `bias=[0.5]`: `acc = 2*3 = 6`,
`z = 6.5`. `u = 0.7978845608 * (6.5 + 0.044715*6.5^3) ≈ 8.31`,
`tanh(u) ≈ 1.0` at that magnitude, so `g ≈ 0.5 * 6.5 * 2.0 = 6.5` --
GELU is nearly identity for large positive inputs.

## What the gate checks

`check.py` runs the kernel over a fixed `4x8` `A`, `8x4` `B`, and
length-4 `bias`. It checks `rel_err <= 1e-6` against a `numpy` oracle
using the identical tanh-approximation formula, `transactions <= 19`,
and `cycles <= 3900`. The reference measures `18` transactions and
`3665` cycles. A correct-but-unfused version -- writing the raw `z =
acc + bias[col]` to `out` first, then reading it back before computing
and writing the final GELU value -- still passes `rel_err` (it computes
the same math) but measures `20` transactions and `4069` cycles, over
both performance gates, from the extra round trip through global memory
that never needed to happen.
