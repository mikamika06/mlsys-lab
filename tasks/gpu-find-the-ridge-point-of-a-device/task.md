## Context

Every device's roofline model has one defining number: the **ridge
point** — the arithmetic intensity (FLOP per byte) at which the
memory-bandwidth line and the peak-compute line cross. Below it, a
kernel is memory-bound (moving data is the bottleneck); at or above it,
the kernel is compute-bound (arithmetic is the bottleneck). It's a
property of the *hardware alone*, fixed the moment you know the device's
peak FLOP/s and peak bytes/s:

$$\text{ridge} = \frac{\text{peak\_flops}}{\text{peak\_bw}}$$

A device with a high ridge point demands very reuse-heavy kernels (like a
well-tiled GEMM) to ever become compute-bound; a low ridge point means
even middling arithmetic intensity is enough.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void ridge_point(float* out, const float* peak_flops, const float* peak_bw, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out[i] = peak_flops[i] / peak_bw[i]`.

## Example

`peak_flops = 16e12` (16 TFLOP/s), `peak_bw = 2e12` (2 TB/s): ridge point
`= 8.0` FLOP/byte. A kernel with arithmetic intensity `4.0` is
memory-bound on this device; one with intensity `16.0` is compute-bound.

## What the gate checks

`max_abs_err <= 1e-6` on 5 fixed `(peak_flops, peak_bw)` device specs,
against a numpy oracle. Dividing the operands the wrong way round
(`peak_bw / peak_flops`, which gives byte/FLOP instead of FLOP/byte) or
mixing up which input array is which produces numbers off by orders of
magnitude and fails every one of the 5 fixed specs.
