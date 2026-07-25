## Context

An **inclusive scan** (prefix sum) turns `[a0, a1, a2, ...]` into
`[a0, a0+a1, a0+a1+a2, ...]`. The **Hillis-Steele** algorithm computes
it in $\log_2 n$ doubling steps instead of $n$ sequential additions:
at step with `offset` $= 1, 2, 4, \dots$, every element at index
`tid >= offset` adds in the element `offset` slots below it:

$$
sdata[tid] \mathrel{+}= sdata[tid - \text{offset}]
$$

After $\log_2 n$ steps, each element has accumulated contributions from
every element below it. The subtlety is *within* one step: thread
`tid`'s read of `sdata[tid - offset]` must see **last step's** value,
not a value some other thread already updated *this* step. Doing the
read and the write back-to-back with no barrier between them is a race
of exactly this kind -- the fix is to have every thread read into a
local register first, `__syncthreads()`, and only then write, so the
read side of the whole step finishes before the write side of any
thread begins.

## Task

Implement:

```cuda
__global__ void inclusive_scan(float* out, const float* in, int n);
```

One thread per element (`n` a power of 2; this task fixes `n = 8`):

1. Load `sdata[tid] = in[tid]` into `__shared__ float sdata[8]`, then
   `__syncthreads()`.
2. For `offset = 1, 2, 4, ...` while `offset < n`:
   - if `tid >= offset`, read `val = sdata[tid - offset]` into a local
     register; else `val = 0`.
   - `__syncthreads()`.
   - if `tid >= offset`, `sdata[tid] += val`.
   - `__syncthreads()`.
3. `out[tid] = sdata[tid]`.

## Example

For `n = 4`, `in = [1, 2, 3, 4]`: step `offset=1`:
`sdata = [1, 1+2, 2+3, 3+4] = [1, 3, 5, 7]`. Step `offset=2`:
`sdata = [1, 3, 5+1, 7+3] = [1, 3, 6, 10]`. Final: `[1, 3, 6, 10]` --
the running totals `1, 1+2, 1+2+3, 1+2+3+4`.

## What the gate checks

`check.py` runs the kernel over 8 fixed random values and checks
`max_abs_err <= 1e-9` against `numpy.cumsum`. Doing the read and the
write of a step without separating them by a barrier lets a thread pick
up a neighbor's already-updated value instead of last step's,
double-counting some elements and under-counting others -- a wrong,
but exactly reproducible, final prefix sum.
