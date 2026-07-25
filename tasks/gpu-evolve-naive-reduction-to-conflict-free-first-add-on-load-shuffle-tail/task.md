## Context

Mark Harris's classic "Optimizing Parallel Reduction in CUDA" walks a sum
reduction through a ladder of fixes to the same naive kernel. Your
starting point is the naive one: one thread per element, "interleaved
addressing" (`if (tid % (2*stride) == 0)`), and every single halving —
all the way from 256 elements down to 1 — going through `__shared__`
memory with a `__syncthreads()` after each step. It's correct. It's also
needlessly expensive in exactly three ways this task fixes:

1. **First-add-on-load**: half the launched threads never need to touch
   shared memory at all if each surviving thread sums TWO input elements
   (`x[tid]` and `x[tid+128]`) into shared memory before the tree even
   starts — the tree only ever has to reduce 128 values, not 256.
2. **Conflict-free (sequential) addressing**: replace `tid % (2*stride) ==
   0` with a shrinking, contiguous `tid < stride` condition. Every active
   thread's shared index stays contiguous at every step — no interleaved
   bank access, no per-step divergence beyond "am I still active."
3. **Shuffle tail**: once the tree is down to 32 live values (one warp),
   stop going through shared memory + `__syncthreads()` altogether — a
   warp already executes in lockstep, so `__shfl_down_sync` can finish the
   last 5 halvings directly register-to-register.

## Task

Write a CUDA-C kernel:

```cpp
__global__ void reduce_sum(float* out, const float* x, int n);
```

Launched as a single block of 256 threads over `n = 256` elements.

1. `__shared__ float sdata[128];`. Only `tid < 128` does anything: each
   such thread writes `sdata[tid] = x[tid] + x[tid + 128]`.
   `__syncthreads()`.
2. Sequential-addressing halvings: `if (tid < 64) sdata[tid] +=
   sdata[tid + 64];` `__syncthreads();` then the same for `tid < 32`
   against `sdata[tid + 32]`. `__syncthreads()`.
3. For `tid < 32`: load `val = sdata[tid]`, then `val +=
   __shfl_down_sync(0xffffffff, val, d)` for `d` in `16, 8, 4, 2, 1` (in
   that order). Thread `0` writes `out[0] = val`.

## Example

On a fixed 256-element random fixture, both the naive starter and the
optimized reference print the exact same sum — correctness never
changes. Only the simulator's shared-memory traffic and cycle estimate
do:

```
naive:     smem_waves=150  cycles=6686
optimized: smem_waves=14   cycles=3704
```

Roughly $10.7\times$ fewer shared-memory waves and $1.8\times$ fewer
modeled cycles, for identical output.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over a fixed 256-element fixture, requiring `max_abs_err <=
1e-9` (the sum itself) AND `smem_waves <= 20` AND `cycles <= 4500` against
the simulator's own measurements — not a comparison to any other file.
The naive starter gets the sum exactly right (`max_abs_err = 0`) but
reports `smem_waves = 150` and `cycles = 6686`, both far past the
thresholds: applying only one or two rungs of the ladder (say,
first-add-on-load without the shuffle tail) still leaves shared-memory
traffic well above `20` waves, so all three optimizations are needed
together to clear both gates.
