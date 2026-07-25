## Context

Matrix multiplication is the textbook case for cache blocking because it
has enormous *reuse*: computing `C[i][j] = sum_k A[i][k] * B[k][j]` for a
whole output tile reads the same rows of `A` and the same columns of `B`
over and over, once per neighboring output element. Whether that reuse
turns into cache hits depends entirely on whether the data being reused is
still resident by the time it's needed again — and that depends on how much
*other* data got touched in between.

A single level of blocking helps one level of cache. Real machines have
several: blocking only for L1 lets the L2-sized surrounding neighborhood
get evicted before it's revisited, and blocking only for L2 leaves L1 doing
no better than an unblocked loop within each L2 tile. **Two-level tiling**
nests both: a large L2-sized tile keeps a big chunk of the working set from
spilling to memory, and a small L1-sized tile *inside* it keeps the
innermost, most-reused sliver resident in the fastest cache.

## Task

Implement, in `solve.cpp`:

```cpp
void matmul_two_level_tiled(const double* A, const double* B, double* C,
                             int M, int N, int K, int lda, int ldb, int ldc);
```

Split the output `(i, j)` space into `32x32` L2 tiles, and each L2 tile into
`8x8` L1 tiles. For one L1 tile, loop `k` from `0` to `K-1` on the
**outside** and `(i, j)` on the **inside**: for each `k`, `touch(&A[i*lda+k])`
once per row `i` in the tile and reuse that value for every `j` in the tile
(a real load stays in a register, it isn't re-fetched per `j`);
`touch(&B[k*ldb+j])` once per `(k, j)`. Accumulate into a local
`double tile_acc[8][8]` (never `touch()`ed — it's assumed to live in
registers) and write it into `C` only after the full `k`-sweep. Clip every
tile's bounds to the matrix edges; don't skip or overrun.

## Example

With `M=N=K=64` and leading dimensions padded to `65` (avoiding an unrelated
aliasing confound), the driver (`main.cpp`, fixed) reports:

```
M=64 N=64 K=64 lda=65 ldb=65 ldc=65
checksum=5975113.7281
l1_misses=39680 l2_misses=2108
```

`checksum` (the sum of every entry of `C`) never depends on tiling — only
the miss counts do. The most obvious GEMM implementation, three plain
nested loops (`i`, `j`, `k`, one scalar accumulator per output element, no
tiling), computes the *same* `checksum=5975113.7281` but pays
`l1_misses=298417` and `l2_misses=12674` — roughly 7.5x more L1 misses and
6x more L2 misses for identical arithmetic, purely from letting each
`(i, j)` dot product sweep all 64 values of `k` (and the resulting
64-line-spanning stride through `B`) independently, with nothing shared
between neighboring outputs still resident when it's their turn.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output — `checksum=` AND
`l1_misses=`/`l2_misses=`, not just one of them — to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Tiling only
the `(i, j)` loops without also moving `k` outside the `(i, j)` loop inside
each tile (i.e. keeping a separate scalar accumulator and a full `k`-sweep
per output element, just inside smaller `(i, j)` blocks) still gets the sum
right but barely dents `l1_misses`, since each individual dot product still
strides through 64 far-apart `B` elements on its own; only reordering `k`
outside `(i, j)` so the tile's `A`/`B` lines get reused across neighboring
outputs reproduces the reference's low miss counts.
