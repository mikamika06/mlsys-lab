## Context

A kernel launch has a FIXED total thread count: `gridDim.x * blockDim.x`.
Real workloads rarely size `n` to match that exactly — often on purpose,
since re-launching with a bigger grid for every different `n` is wasteful,
and there are hardware caps on how many blocks a single launch can hold.
The standard fix is a **grid-stride loop**: instead of each thread doing
exactly one element and stopping, thread `i` handles element `i`, then
`i + stride`, then `i + 2*stride`, ... (`stride = gridDim.x * blockDim.x`)
until it runs past `n`. However small the launch grid is relative to `n`,
every element still gets covered — just by fewer threads doing more work
each.

Drop the loop — write `if (i < n)` and stop — and every element from
`gridDim.x * blockDim.x` onward is silently never touched. The bug is
invisible whenever a test happens to launch enough threads to cover the
whole array; it only shows up once `n` genuinely exceeds the launch size.

## Task

Your starting point in `solve.cu` computes `out[i] = 2*in[i] + 1` for a
single element per thread, with no loop:

```cpp
__global__ void grid_stride_scale(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 2.0f * in[i] + 1.0f;
    }
}
```

Fix it into a proper grid-stride loop: compute `stride = gridDim.x *
blockDim.x`, then `for (; i < n; i += stride) out[i] = 2.0f * in[i] +
1.0f;` — every thread keeps processing elements `stride` apart until it
runs out of array.

## Example

The driver launches `grid = 2` blocks of `block = 64` threads — `128`
total threads — over `n = 300` elements. A correctly looping thread `5`
handles indices `5, 133, 261`; the broken, loop-less version only ever
touches index `5`. Every index from `128` to `299` — 172 of the 300
elements, well over half the array — is left at its `-1.0` sentinel by
the broken version.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over the fixed `n = 300` fixture with only `128` total
threads, requiring `max_abs_err <= 1e-9` against `2*in + 1` computed in
numpy over the FULL 300-element array. The single-pass starter gets the
first 128 elements exactly right and leaves the remaining 172 untouched at
`-1.0`, which is nowhere close to their correct values and fails the gate
immediately.
