## Context

A transformer's residual block runs dropout, a residual add, and a
LayerNorm back to back:

$$y = \text{LayerNorm}\bigl(\text{dropout}(x) + \text{residual}\bigr)$$

Run as three separate kernels, that's three full passes over the row:
dropout writes its result to memory, the add reads it back and writes
again, LayerNorm reads that back for its mean/variance pass and writes
the final output. **Fusing** all three into one kernel — one block per
row, every intermediate value kept in registers or `__shared__` memory —
cuts that to a single pass: read `x` and `residual` once, write `out`
once.

This language subset has no RNG builtin, so "fixed-seed dropout" here
means a **deterministic arithmetic hash** of the flat element index and a
seed, not a stateful random-number generator — the exact same formula
produces the exact same mask, every time, on any machine. For a
$d$-feature row's flat index $i$: `h = (i*31 + seed*7 + 11) % 100`; the
element is dropped (zeroed) if `h < dropout_p * 100`, otherwise kept and
rescaled by $1/(1-\text{dropout\_p})$ (inverted dropout, so no rescaling
is needed at inference time).

## Task

Write a CUDA-C kernel, one block per row (`d` threads per block, thread
`j` handles feature `j` of row `blockIdx.x`):

```cpp
__global__ void fused_block(float* out, const float* x, const float* residual,
                             const float* gamma, const float* beta,
                             int d, float dropout_p, float eps, int seed);
```

1. `idx = row * d + j`. Compute the dropout keep/drop decision from `h =
   (idx*31 + seed*7 + 11) % 100` as above, and `v = x[idx] * keep * scale
   + residual[idx]` where `scale = 1/(1-dropout_p)`. Store `v` into
   `__shared__ float buf[64]` at `buf[j]`. `__syncthreads()`.
2. Thread `0` reduces `buf[0..d)` into the row mean (`__shared__ float
   mean_s[1]`). `__syncthreads()`.
3. Every thread computes `dev = buf[j] - mean_s[0]` and stores `dev*dev`
   into `__shared__ float sqbuf[64]`. `__syncthreads()`. Thread `0`
   reduces `sqbuf[0..d)` into the row variance (`__shared__ float
   var_s[1]`). `__syncthreads()`.
4. `out[idx] = dev / sqrtf(var_s[0] + eps) * gamma[j] + beta[j]`.

## Example

The driver runs this over $R = 4$ rows of $D = 16$ features
(`dropout_p=0.3, eps=1e-5, seed=7`), with fixed random `x`, `residual`,
`gamma`, `beta`. The reference kernel's output matches a numpy
implementation of the exact same formula (same hash, same LayerNorm
statistics) to within floating-point rounding —
`max_abs_err ≈ 4.4e-16`, not exactly `0`, because both the kernel and the
oracle do the same chain of `float`/`float64` arithmetic in a slightly
different order.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over the fixed $4 \times 16$ fixture, requiring
`max_abs_err <= 1e-6` against the numpy oracle (which replicates the
identical hash formula — there is nothing "random" to guess, only
arithmetic to reproduce exactly). Using a DIFFERENT hash formula, a
different dropout threshold direction (`h >= thresh` instead of `h <
thresh`), or LayerNorm statistics computed over the wrong axis, all
produce numbers wildly different from the reference and fail immediately.
The empty starter never writes `out` and fails trivially.
