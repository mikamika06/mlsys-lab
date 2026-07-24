## Context

The classic first GPU kernel: elementwise vector addition. Each thread owns one output element. Using contiguous per-thread indices yields coalesced global accesses -- a single 128-byte transaction serves all 32 lanes of a warp, instead of one transaction per lane.

## Task

Write the CUDA-C kernel `vecAdd(float* out, const float* a, const float* b, int n)`.

**Global-memory layout** (three arrays of length $n$ each): `a[0..n-1]`, `b[0..n-1]`, `out[0..n-1]`.

For each thread `i = blockIdx.x * blockDim.x + threadIdx.x` with `i < n`: `out[i] = a[i] + b[i]`. Threads with `i >= n` must do nothing.

## Example

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = a[i] + b[i];
}
```

## What the gate checks

The grader parses your kernel with the real CUDA-C interpreter and launches it on the software GPU over a grid covering `N = 256` elements with block size 64, then compares `out` against `a + b` computed with NumPy.

| Metric | Condition | Meaning |
|---|---|---|
| `max_abs_err` | $\le 10^{-9}$ | Output matches `a + b` exactly |
| `transactions` | $\le 300$ | Global-memory access is coalesced |
