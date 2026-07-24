## Context

Global memory bandwidth is precious on a GPU, so kernels often process
several elements per thread to amortize per-access overhead — e.g. each
thread handles 4 consecutive elements instead of 1 (the scalar stand-in for
a real `float4` vectorized load/store, which this CUDA-C subset doesn't
support as a type). This works great when the array length `n` is a
multiple of the group width (4 here). It breaks the moment it isn't: the
*last* thread's group of 4 can run past the end of the array, reading and
writing out of bounds.

## Task

Fix the kernel

```cuda
__global__ void vector_copy_kernel(float* out, const float* in, int n);
```

so it copies `in[0..n)` to `out[0..n)`, with each thread `tid` handling the
4-element group starting at `base = tid * 4`, **without ever touching
`in[i]` or `out[i]` for `i >= n`**. Guard each of the 4 element accesses
individually against `n` — the last thread's group may have anywhere from 1
to 4 elements actually in bounds.

## Example

For `n = 13`, `blockDim.x = 4`, one block (4 threads):

```
thread 0: base=0  -> writes out[0..4)   (all 4 in bounds)
thread 1: base=4  -> writes out[4..8)   (all 4 in bounds)
thread 2: base=8  -> writes out[8..12)  (all 4 in bounds)
thread 3: base=12 -> writes out[12]     (in bounds); out[13],out[14],out[15]
                      are ALL past the end of the array (n=13) and must be
                      skipped
```

## What the gate checks

The grader (`check.py`) launches your kernel on the deterministic software
GPU (`arena.cuda_sim.GPU`) with `n = 13` (deliberately not a multiple of 4)
and one block of 4 threads, then compares `out[0..n)` against `in[0..n)`:

$$ \mathrm{max\_abs\_err} = \max_i |\,\mathrm{out}[i] - \mathrm{in}[i]\,| \le 10^{-12} $$

An unguarded kernel writes to `out[13]`, `out[14]`, `out[15]` — indices that
fall outside the simulator's allocated memory entirely — which raises a
real out-of-bounds error from the simulator. That is caught and reported as
$\mathrm{max\_abs\_err} = \infty$, failing the gate immediately, exactly like
an out-of-bounds write can crash a real kernel.
