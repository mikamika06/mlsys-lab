## Context

RMSNorm normalizes each row by its root-mean-square instead of its
standard deviation — no mean-subtraction, one reduction instead of two:

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma_i$$

`mean(x^2)` is a single reduction over the row, shared by every element
in it: every thread needs the *same* final value, so the reduction has to
finish (and its result be visible to the whole block) before any thread
computes its own output. `eps` guards against division by zero when a
row is exactly (or nearly) all zeros.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void rmsnorm_forward(float* out, const float* x, const float* gamma, float eps, int n);
```

One block of `n=32` threads, one row. Compute `mean(x^2)` with a
sequential-addressing tree reduction in `__shared__` memory (square
`x[tid]` into `sdata[tid]`, then reduce with `stride = blockDim.x/2` down
to `1`, barrier every step). Then every thread computes
`out[tid] = (x[tid] / sqrtf(sdata[0]/n + eps)) * gamma[tid]`.

## Example

`x = [3, 4]` (2-element toy row, `gamma = [1, 1]`, `eps = 0`):
`mean(x^2) = (9+16)/2 = 12.5`, `rms = sqrt(12.5) ≈ 3.536`, so
`out ≈ [0.849, 1.131]`.

## What the gate checks

`max_abs_err <= 1e-6` on a fixed 32-element row against a numpy oracle.
Reading `sdata[0]` before every thread's contribution has been reduced in
(a missing or misplaced `__syncthreads()`), forgetting to divide by `n`
before the square root, or applying `gamma` before dividing by `rms`
instead of after, all change the printed row.
