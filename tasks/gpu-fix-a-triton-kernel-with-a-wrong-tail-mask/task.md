## Context

A GPU launch always starts a whole number of blocks, and a block always
starts a fixed number of threads — so unless the data size happens to be an
exact multiple of `blockDim.x * gridDim.x`, the very last block has some
threads whose global index runs *past* the real data. Every such thread
must be masked off before it touches memory: it has no valid input to read
and nowhere valid to write.

The bug this task is about is subtle because it *looks* like a bounds
check: `if (i < blockDim.x * gridDim.x)`. That comparison is not wrong
syntax, and it never crashes — but `i` is a `blockIdx.x * blockDim.x +
threadIdx.x` value produced by the very same launch, so it is *always* less
than `blockDim.x * gridDim.x` by construction. The condition is trivially
true for every thread, which means it excludes nothing: it is not a tail
mask at all, just a bounds check against a bound no thread can ever exceed.
The comparison needs to be against the real data size, `n`, not the launch
geometry.

## Task

Fix the kernel in `solve.cu`:

```cuda
__global__ void scale_masked(float* out, const float* in, int n, float s);
```

`n` is not a multiple of `blockDim.x * gridDim.x`, so the last block has
threads with `i >= n`. Compute `i = blockIdx.x * blockDim.x + threadIdx.x`
and write `out[i] = s * in[i]` **only** when `i < n`; threads with
`i >= n` must not read or write anything at all.

## Example

The grader launches `blockDim.x = 32`, `gridDim.x = 4` (128 threads total)
over `n = 100` real elements — 28 threads in the last block are tail
threads. `out` is pre-filled with a sentinel value before the launch. A
correctly masked kernel leaves indices `100..127` exactly as that
sentinel; the buggy version above overwrites all of them with `s * in[i]`
computed from whatever happens to be sitting in `in[100..127]`, which the
grader treats as visible, comparable garbage — not a crash, just silently
wrong output past index 99.

## What the gate checks

`check.py` builds the fixture, parses `solve.cu`, and runs `scale_masked`
on the software GPU (`arena.cuda_sim.GPU`) with a 4-block, 32-thread
launch over `n = 100`. It requires `max_abs_err == 0.0` against a reference
that fills indices `[0, 100)` with `s * in[i]` and leaves indices
`[100, 128)` exactly at the pre-launch sentinel — comparing the **whole**
128-element output buffer, not just the first 100 elements, is what makes
the trivially-true mask's leak visible at all.
