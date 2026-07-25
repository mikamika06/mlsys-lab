## Context

The roofline model bounds a kernel's attainable throughput by two lines:
a memory-bandwidth line that grows with arithmetic intensity
($\text{AI} \times \text{peak\_bw}$) and a flat peak-compute line
($\text{peak\_flops}$). The kernel can never exceed whichever line is
lower at its own AI — the achievable throughput is simply the minimum of
the two:

$$\text{attainable} = \min(\text{peak\_flops},\; \text{AI} \times \text{peak\_bw})$$

Below the ridge point, the bandwidth line is the binding constraint
(memory-bound); at or above it, the compute line is (compute-bound).

## Task

Implement, in real CUDA-C:

```cuda
__global__ void attainable_flops(float* out, const float* peak_flops, const float* peak_bw,
                                  const float* ai, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out[i] = fminf(peak_flops[i], ai[i] * peak_bw[i])`.

## Example

`peak_flops=16e12, peak_bw=2e12`: at `ai=2.0` (bandwidth-heavy), the
bandwidth line gives `2.0 * 2e12 = 4e12`, below `peak_flops` — attainable
is `4e12` (memory-bound). At `ai=16.0`, the bandwidth line gives `32e12`,
above `peak_flops` — attainable caps at `16e12` (compute-bound). At
`ai=8.0` exactly (the ridge point), both lines agree: `16e12` either way.

## What the gate checks

`max_abs_err <= 1e-3` on 5 fixed `(peak_flops, peak_bw, ai)` triples,
including the exact ridge-point case (`ai=8.0` for a device whose ridge
point is `8.0`) where `min` must still return the shared value. Using
`fmaxf` instead of `fminf`, or dividing instead of multiplying `ai` by
`peak_bw`, produces numbers wildly off from the fixed references.
