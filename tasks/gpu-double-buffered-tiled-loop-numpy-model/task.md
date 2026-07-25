## Context

A tiled matmul loop that streams K in chunks has an obvious-looking
shortcut: load a tile into `__shared__` memory, sync, compute with it, sync,
load the next tile into the *same* buffer, sync, compute, ... Each tile's
load has nothing to do mathematically with the previous tile's compute step
— they touch completely different iterations of `k` — so it seems wasteful
to serialize them behind two separate barriers when they could, in
principle, happen side by side.

**Double buffering** is how you actually get that overlap safely: keep
*two* shared-memory buffers instead of one, and while the current tile
(buffer A) is being consumed by the compute loop, prefetch the *next* tile
into buffer B. No barrier is needed between the prefetch stores and the
compute reads, because they never touch the same memory. Only once both are
done do you sync and switch which buffer is "current."

Try to get the same overlap with a *single* buffer and you don't just lose
the performance benefit — you get the wrong answer. Nothing here executes
in true lockstep at the individual-statement level: between two
`__syncthreads()` calls, each thread runs its whole segment of code before
the next thread's turn. If "prefetch the next tile" and "compute with the
current tile" share one buffer and aren't separated by a barrier, some
threads finish overwriting it with the next tile's data before other
threads have read the current tile's data out of it for their own compute.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void tiled_matmul_double_buffered(float* C, const float* A, const float* B,
                                              int M, int N, int K, int tile_k);
```

One block, `M*N` threads (`M = N = 8`), `tid -> (row = tid/N, col = tid%N)`.
`K = 16`, `tile_k = 8` — exactly two K-tiles. Load tile 0 of `A`/`B` into a
`__shared__` buffer pair, sync, then: prefetch tile 1 into a **second,
separate** buffer pair (no sync needed before this — it doesn't touch the
first pair), accumulate tile 0's partial sum from the first pair, sync
(now tile 1 is guaranteed fully loaded and tile 0's readers are done), then
accumulate tile 1's partial sum from the second pair. Write the total to
`C[row*N + col]`.

## Example

The grader computes the reference `C` with an explicit Python loop that
sums tile 0's contribution, then tile 1's, in that exact order — so a
correct double-buffered kernel matches it **exactly**, `max_abs_err = 0.0`.
A kernel that attempts the identical overlap (prefetch tile 1 positioned
right after the first sync, no barrier before the tile-0 compute loop) but
reuses **one** shared buffer for both tiles instead of two computes
`max_abs_err = 2.7781986...` on this fixture: some threads' tile-1
prefetch stores land before other threads have finished reading tile 0 out
of that same buffer, so several outputs end up as a scramble of tile-0 and
tile-1 contributions instead of the sum of both.

## What the gate checks

`check.py` builds the fixture, parses `solve.cu`, and runs
`tiled_matmul_double_buffered` on the software GPU
(`arena.cuda_sim.GPU`) with a 1-block, 64-thread launch. It requires
`max_abs_err == 0.0` — exact, not approximate, since both tiles' true
contributions are ordinary in-range floating-point sums with nothing
numerically delicate about them; any discrepancy at all means the overlap
was implemented unsafely. A "fix" that keeps one shared buffer but adds a
second `__syncthreads()` between the prefetch and the tile-0 compute loop
would also reach `0.0` — but that reintroduces exactly the serialization
double buffering exists to avoid, which is why the reference is built with
two separate buffer pairs instead.
