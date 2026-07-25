## Context

A transformer block computes `LayerNorm(x + residual)` constantly. The
unfused way runs it as two kernels: one that adds `x + residual` and
writes the sum to a fresh array in global memory, and a second that reads
that array back, computes the mean/variance, and writes the normalized
result. That round trip — write the sum out, then immediately read it
back in — is pure waste: nothing outside the LayerNorm computation ever
needs that intermediate value.

**Kernel fusion** means never letting that intermediate value leave
registers: recompute `x[i] + residual[i]` directly, in both places it's
needed (once while accumulating statistics, once while normalizing),
instead of storing it anywhere. Fusing a kernel means changing what
happens to *memory*, not what happens to the *math* — the numeric result
is identical either way.

## Task

Implement

```cpp
__global__ void fused_layernorm_residual(float* out, const float* x, const float* residual,
                                          const float* gamma, const float* beta,
                                          int N, int D, float eps);
```

One thread per row (`i = threadIdx.x`, `N` rows of `D` features each).
For row `i`:

1. Pass 1: accumulate `sum` and `sumsq` of `x[i*D+d] + residual[i*D+d]`
   for `d` in `[0, D)`. Compute `mean = sum/D`,
   `var = sumsq/D - mean*mean`, `inv_std = 1/sqrt(var + eps)`.
2. Pass 2: for `d` in `[0, D)`, recompute `v = x[i*D+d] + residual[i*D+d]`
   (do **not** read it back from anywhere it was stored), then
   `out[i*D+d] = (v - mean) * inv_std * gamma[d] + beta[d]`.

Never write `x[i*D+d] + residual[i*D+d]` to `out` (or anywhere else) as
an intermediate step and then read it back — recompute it in registers
both times it's needed.

## Example

For row `i` with `D=8`: pass 1 touches `x[i*8 .. i*8+7]` and
`residual[i*8 .. i*8+7]` once each to build `sum`/`sumsq` — 16 reads, 0
writes. Pass 2 touches the same 16 addresses again, plus reads
`gamma[0..7]`/`beta[0..7]` and writes `out[i*8..i*8+7]` — but at no point
does row `i`'s sum ever get written anywhere and read back.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend, runs it on a
fixed 32-row, 8-feature random input, and compares `out` against
`LayerNorm(x + residual) * gamma + beta` computed in numpy. It also runs
an UNFUSED two-kernel pipeline itself (via the simulator's native Thread
API: one kernel that writes `x + residual` to a scratch array, a second
that reads it back and normalizes) to measure a baseline transaction
count, and requires the candidate's ratio to that baseline to be
genuinely lower:

$$
\mathrm{max\_abs\_err} \le 10^{-6}, \qquad \mathrm{transaction\_ratio} \le 0.9
$$

On this fixture the reference measures `transaction_ratio = 0.84` — 16%
fewer transactions than the unfused baseline, purely from never writing
the intermediate sum to memory. A single-kernel "fusion" that still
writes `x + residual` into `out` first and reads it back before
normalizing (technically one kernel launch, but still round-tripping the
intermediate through memory) measures the exact **same** transaction
count as the true unfused baseline — `transaction_ratio = 1.0` — because
the gate counts memory traffic, not kernel launches.
