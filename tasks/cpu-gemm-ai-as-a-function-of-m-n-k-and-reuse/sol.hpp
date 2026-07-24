#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A GEMM C[M,N] = A[M,K] x B[K,N], computed with square output TILES of
// size `tile` x `tile` (tile divides both M and N): each tile accumulates
// its `tile*tile` outputs over the full K-deep reduction before moving on.
// Within one tile, each element of the A-panel it uses is read once and
// then reused for all `tile` columns of that tile; each element of the
// B-panel it uses is read once and reused for all `tile` rows of that
// tile.
//
// FLOPs (multiply + add counted separately):
//     flops = 2 * M * N * K
//
// Bytes moved (elem_bytes per element):
//   - A is re-read once per column-tile: a (M x K) pass happens (N / tile)
//     times -> M*K*(N/tile) elements.
//   - B is re-read once per row-tile: a (K x N) pass happens (M / tile)
//     times -> K*N*(M/tile) elements.
//   - C is written once: M*N elements.
//     bytes = elem_bytes * ( M*K*(N/tile) + K*N*(M/tile) + M*N )
//
// Return the arithmetic intensity (FLOP per byte):
//     ai = flops / bytes
//
// `tile == 1` recovers the fully-naive, no-reuse baseline; as `tile`
// grows, AI grows (fewer redundant re-reads of A and B per FLOP done).
// All inputs are positive; `tile` always evenly divides both M and N in
// the driver's scenarios.
// ============================================================================
double gemm_arithmetic_intensity(long M, long N, long K, long tile, long elem_bytes);
