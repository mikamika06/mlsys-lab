## Context

A tiled GEMM loads a `TILE x TILE` block of `A` and `B` into shared
memory, computes a partial dot product from it, then moves on to the
next `K`-tile. Two separate concerns stack on top of that basic idea:

**Software pipelining (double buffering)**: instead of load-then-compute,
load-then-compute, load-then-compute (stalling on every load), keep TWO
copies of each tile. While iteration `kt` computes from buffer
`kt % 2`, it also prefetches K-tile `kt+1` into the OTHER buffer,
`(kt+1) % 2`. Get the buffer index wrong — compute from the buffer that's
*currently being overwritten* instead of the one already fully loaded —
and you silently read garbage or stale-plus-half-fresh data: still a
*number*, just the wrong one.

**Bank-conflict-free layout**: this kernel assigns one **column** of the
output tile per warp (`col = threadIdx.x / 32`, so all 32 lanes of a warp
share one `col` and sweep `row = threadIdx.x % 32`). That means every
compute step reads `As[row * stride + e]` at 32 *different* `row`s
simultaneously. With `stride = TILE` (32), `row * 32` is always a
multiple of 32 — every one of those 32 different addresses lands in
**bank 0**, a 32-way conflict on every single step. Pad the stride to
`TILE + 1` (33) and `row * 33 mod 32 == row` — the 32 lanes now spread
across all 32 banks.

## Task

Implement

```cpp
__global__ void gemm_tile_dbuf(float* C, const float* A, const float* B, int M, int N, int K);
```

for `M = N = 32`, `K = 64` (two 32-wide K-tiles), launched as a single
block of 1024 threads, `col = threadIdx.x / 32`, `row = threadIdx.x % 32`.
Use two `__shared__` arrays (`As[2112]`, `Bs[2112]` — 2 buffers of
`32 * 33` padded words each):

1. Prologue: load K-tile 0 into buffer 0 —
   `As[0*1056 + row*33 + col] = A[row*K + 0*32 + col];` and the matching
   `Bs` line from `B` — then `__syncthreads();`.
2. For `kt` in `0, 1`: let `buf = kt % 2`, `nbuf = (kt+1) % 2`. If
   `kt + 1 < 2`, prefetch K-tile `kt+1` into `nbuf` (same indexing
   pattern as the prologue, offset by `kt+1`). Then accumulate
   `acc += As[buf*1056 + row*33 + e] * Bs[buf*1056 + e*33 + col]` for
   `e` in `[0, 32)`. Then `__syncthreads();` (every iteration, whether or
   not a prefetch happened).
3. `C[row*N + col] = acc;`.

## Example

At `kt=0`: `buf=0` (already loaded by the prologue), `nbuf=1` gets K-tile
1 prefetched *while* the loop is about to accumulate from `buf=0`. The
`__syncthreads()` at the end of `kt=0` guarantees `nbuf=1`'s prefetch has
finished before `kt=1` reads it as `buf=1`.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend, runs it on a
fixed `32x64` / `64x32` random input, and compares `C` against numpy's
own `A @ B`. It also runs the grader's OWN padded, double-buffered
kernel (written directly against the simulator's native Thread API, not
compiled from a `.cu` file) to measure a conflict-free lower bound on
shared-memory traffic, and requires the candidate's ratio to that bound
stay low:

$$
\mathrm{max\_abs\_err} \le 10^{-6}, \qquad \mathrm{smem\_wave\_ratio} \le 1.5
$$

On this fixture the reference hits the floor exactly
(`smem_wave_ratio = 1.0`). Dropping the `+1` padding (`stride = 32`
instead of `33`) still computes the exact right answer but measures
`smem_wave_ratio ≈ 17` — a 32-way bank conflict on every one of the 64
compute steps, because `row * 32` collapses onto bank 0 for every `row`
simultaneously. Swapping which buffer the accumulation reads from (`nbuf`
instead of `buf` — computing from the tile that's still being written
instead of the one already resident) keeps `smem_wave_ratio = 1.0` but
produces a wrong sum: getting the layout right and getting the pipeline
sequencing right are two independent ways to fail here, and the two
gates catch them separately. (This simulator's cost model is a simple
additive sum, not a pipeline model, so it cannot measure double
buffering's real hardware benefit — hiding load latency behind compute;
only whether the buffer-swap logic is *correct*.)
