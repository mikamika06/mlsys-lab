## Context

A naive matmul thread computing `C[row][col]` reads its own entire
`K`-length row of `A` and column of `B` straight from global memory (DRAM)
— but every other thread in the same block-row re-reads that exact same
row of `A` all over again, and every thread in the same block-column
re-reads the same column of `B`. Nothing is shared: `K` global loads per
array, per thread, with massive redundancy across threads.

**Tiling** breaks the matmul into `T x T` chunks along all three
dimensions. For each `T`-wide step along `K`, the whole block
*cooperatively* loads one `T x T` tile of `A` and one of `B` into
`__shared__` memory — each element pulled from DRAM exactly once, by
exactly one thread — then every thread in the block reuses that same
resident tile `T` times (once per `k` in the tile) to advance its own
accumulator. The redundant re-reads move from DRAM (global memory,
hundreds of cycles) to shared memory (tens of cycles), and each element
that used to be fetched from DRAM `T` times now gets fetched once.

## Task

Implement

```cuda
__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);
```

for `16x16` tiles. This simulator's CUDA-C frontend is 1D-only (grid and
block are plain ints; `.y` always reads as a constant), so derive the
2D tile coordinates by hand from the linear `threadIdx.x` /
`blockIdx.x`:

```
tx = threadIdx.x % 16,  ty = threadIdx.x / 16
blocksPerRow = N / 16
blockCol = blockIdx.x % blocksPerRow,  blockRow = blockIdx.x / blocksPerRow
row = blockRow*16 + ty,  col = blockCol*16 + tx
```

Then for `kt = 0 .. K/16 - 1`: cooperatively load
`As[ty*16+tx] = A[row*K + kt*16+tx]` and
`Bs[ty*16+tx] = B[(kt*16+ty)*N + col]` into `__shared__ float As[256]`,
`Bs[256]`; `__syncthreads()`; accumulate
`acc += As[ty*16+k] * Bs[k*16+tx]` for `k` in `[0, 16)`; `__syncthreads()`.
Finally `C[row*N+col] = acc`.

## Example

For a single `16x16` tile (`M=N=K=16`, one tile-step): the block loads
`256` elements of `A` and `256` of `B` from DRAM — once each, total `512`
DRAM reads — then every one of the `256` threads does `16` shared-memory
reads of each array (`8192` shared reads total) to finish its dot
product. A naive kernel would need `256 * 16 = 4096` DRAM reads of `A`
alone for the same tile.

## What the gate checks

`check.py` seeds a fixed random `32x32 @ 32x32` matmul, launches your
`matmul_tiled` (checked against a numpy `A @ B` oracle), and separately
launches a fixed, always-correct naive kernel (embedded in `check.py`,
never your code) that re-reads `A`/`B` from global memory with no
sharing. It reads both launches' `transactions` (128-byte DRAM segments
touched) and computes `size_ratio = naive_transactions /
tiled_transactions`. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6} \quad \text{and} \quad \mathrm{size\_ratio} \ge 8
$$

The reference measures `tiled_transactions=320` against
`naive_transactions=3136` — `size_ratio=9.8`, most of the way to the
tile's theoretical `16x` reduction (coalescing overhead on partial
segments accounts for the rest). An empty kernel body leaves `C` all
zeros (`max_abs_err` far off) and issues zero global transactions
(`size_ratio=0.0`), failing both gates.
