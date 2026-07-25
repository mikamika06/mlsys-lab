#pragma once

struct MissVector {
    int l1_misses;
    int l2_misses;
};

// ============================================================================
// FIXED two-level cache probe (defined in main.cpp — do not modify).
//   L1: 8 sets, 2 ways, 64-byte lines  -> 1024 bytes (128 doubles) total.
//   L2: 64 sets, 8 ways, 64-byte lines -> 32768 bytes (4096 doubles) total.
// Every touch(p) is checked against L1 first. On an L1 HIT, that's it --
// L2 is not even consulted. On an L1 MISS, L1's miss count goes up and the
// SAME line is checked against L2; if L2 also misses, L2's miss count goes
// up too. Either way the line is inserted into whichever level(s) missed
// (both LRU).
//   cache_reset() empties both levels and zeroes both counters.
//   touch(p) performs one such lookup.
//   miss_vector() returns {l1_misses, l2_misses} since the last reset.
// ============================================================================
void cache_reset();
void touch(const void* p);
MissVector miss_vector();

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// matmul_two_level_tiled: compute C = A * B for an M x K times K x N
// matrix product (A: M x K row-major, leading dimension lda; B: K x N
// row-major, leading dimension ldb; C: M x N row-major, leading dimension
// ldc). C's accumulator tile is assumed to live in registers -- never
// call touch() for it, only for A and B reads.
//
// Use NESTED two-level tiling over the OUTPUT (i, j) space:
//   - split (i, j) into L2_TILE x L2_TILE outer tiles (L2_TILE = 32),
//   - within each outer tile, split into L1_TILE x L1_TILE inner tiles
//     (L1_TILE = 8).
// For one inner tile, loop k OUTERMOST (0 to K-1) and (i, j) innermost:
// for each k, touch(&A[i*lda+k]) ONCE per row i in the tile and reuse
// that value for every j in the tile (a real load would sit in a
// register for all of them, not get re-fetched); touch(&B[k*ldb+j]) once
// per (k, j). Accumulate into a local `tile_acc[L1_TILE][L1_TILE]`
// (never touched) and write it into C only after the full k-sweep.
// (Edges that don't divide evenly by a tile size get a partial tile —
// clip each tile's bounds, don't skip or overrun.) This keeps one inner
// tile's A-rows/B-row-slice resident in L1 across its own k-sweep, and
// one outer tile's full working set resident in L2 across all its inner
// tiles.
// ============================================================================
void matmul_two_level_tiled(const double* A, const double* B, double* C,
                             int M, int N, int K, int lda, int ldb, int ldc);
