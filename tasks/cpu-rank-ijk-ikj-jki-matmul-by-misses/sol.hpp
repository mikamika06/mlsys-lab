#pragma once

// ============================================================================
// FIXED cache probe (defined in main.cpp — do not modify). A small
// deterministic set-associative LRU cache: 64-byte lines, 32 sets, 2-way
// (4096 bytes total capacity).
//   cache_reset()   empties it.
//   touch(addr)     accesses byte address `addr`.
//   cache_misses()  returns misses since the last reset.
// ============================================================================
void cache_reset();
void touch(long byte_addr);
long cache_misses();

// Row-major address helpers for three n x n matrices of 8-byte doubles,
// laid out back-to-back in memory: A first, then B, then C.
inline long a_addr(int n, int i, int k) { return (long)(i * n + k) * 8; }
inline long b_addr(int n, int k, int j) { return (long)n * n * 8 + (long)(k * n + j) * 8; }
inline long c_addr(int n, int i, int j) { return 2L * n * n * 8 + (long)(i * n + j) * 8; }

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Run THREE separate simulated passes of C += A * B over the SAME
// n x n x n index space, one per loop nest below, EACH on its own fresh
// cache (call cache_reset() first). For every (i, j, k) touch
// a_addr(n,i,k), then b_addr(n,k,j), then c_addr(n,i,j) -- in that order,
// EXACTLY ONCE per triple -- while the three nested loops run in the
// stated order:
//
//   "ijk": for i, for j, for k
//   "ikj": for i, for k, for j
//   "jki": for j, for k, for i
//
// After each pass, read cache_misses(). Write the three order labels
// ("ijk", "ikj", "jki" -- each a NUL-terminated string, buffer size 4)
// into out[0], out[1], out[2], SORTED from FEWEST misses to MOST. Break
// ties by the fixed priority ijk < ikj < jki.
// ============================================================================
void rank_matmul_orders(int n, char out[3][4]);
