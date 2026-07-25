## Context

A histogram of $n$ input elements into $B$ bins is embarrassingly parallel
in principle, but every naive implementation funnels all $n$ increments
into the same $B$ global addresses, so most threads collide with some
other thread on the same bin.

The standard fix is **privatization**: each block first accumulates its own
slice of the input into a histogram PRIVATE to that block, living in
`__shared__` memory. Contention there is limited to the (at most 32)
threads of a single block, and only when two of them land on the same bin
-- an ordinary shared-memory `atomicAdd` handles it cheaply. Once every
thread in the block has contributed, exactly one thread per bin does **one**
`atomicAdd` to flush that block's partial count into the global histogram.
Contention at that point is between different blocks' bin-$k$ flushes (all
of them target the same global address), so that flush must also be
atomic -- but there are only (num\_blocks $\times$ $B$) such global atomics
total, versus $n$ if every element atomically updated the global histogram
directly.

## Task

Implement, in `solve.cu`:

```cpp
__global__ void histogram_privatized(const float* input, float* out, int n);
```

- `input` holds `n` values, each an integer bin index in $[0, 8)$ stored as
  a float.
- `out` holds 8 running bin counts (pre-zeroed by the driver, accumulated
  across every block).
- Declare `__shared__ float hist[8];`, zero it (e.g. by the first 8 threads
  of the block), `__syncthreads()`, have every in-range thread
  `atomicAdd(&hist[bin], 1.0f)` its own element's bin, `__syncthreads()`
  again, then have exactly the first 8 threads of the block
  `atomicAdd(&out[tid], hist[tid])` to flush.

## Example

With 128 elements, 8 bins, and 4 blocks of 32 threads, `out[k]` after the
kernel must equal the number of input elements equal to `k` -- identical to
`numpy.bincount(input, minlength=8)`.

## What the gate checks

`max_abs_err` compares `out` against `numpy.bincount` of the same input the
kernel saw -- the histogram must be exactly right.

`races` is the simulator's count of addresses written by more than one
thread where at least one of those writes was non-atomic. The simulator
executes threads one at a time, so a naive non-atomic accumulate (`hist[bin]
= hist[bin] + 1.0f;` or `out[tid] = out[tid] + hist[tid];`) still happens to
sum to the right total in THIS simulator -- `max_abs_err` alone would not
catch it. `races` does: with 32 threads spread over only 8 bins, some
same-block threads are guaranteed to land on the same bin, and every
block's bin-`k` flush targets the same global address as every other
block's, so both accumulation stages have a real, detectable hazard unless
both use `atomicAdd`.
