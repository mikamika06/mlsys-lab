#pragma once

// Cache access hook, DEFINED in main.cpp: touch() records the byte
// address against a small deterministic set-associative LRU cache.
// reset_cache() clears it (fresh state); miss_count() reads the
// cumulative miss count since the last reset. Real hardware cache
// behaviour is not reproducible across machines, so this model is the
// sole source of every miss count.
void touch(long byte_addr);
void reset_cache();
long miss_count();

// Row-major addressing for an N x N float matrix starting at byte address
// `base`: element (r, c) lives at base + (r*N + c)*4.
inline long addr(int N, long base, int r, int c) { return base + (long)(r * N + c) * 4; }

// Multiply C = A*B for an N x N matmul (N a power of two, divisible by
// tile1, tile1 divisible by tile2) using THREE different loop orders over
// the exact same i-j-k index space, each against its own FRESH cache
// (call reset_cache() right before each variant), and write each
// variant's miss_count() right after it finishes into out[0..2]:
//
//   out[0]  NAIVE: flat i-j-k triple loop over the whole N x N x N space,
//           in that order. For every (i, j, k), touch addr(A,i,k), then
//           addr(B,k,j), then addr(C,i,j) -- in that order.
//
//   out[1]  1-LEVEL TILED: block the i-j-k space into tile1 x tile1 x
//           tile1 tiles; within one tile, iterate i, j, k directly, same
//           touch order as the naive case.
//
//   out[2]  2-LEVEL TILED: block the i-j-k space into tile1 x tile1 x
//           tile1 OUTER tiles, and within each outer tile, into a further
//           tile2 x tile2 x tile2 INNER tiles; within one inner tile,
//           iterate i, j, k directly, same touch order.
//
// All three variants must touch exactly the same N*N*N*3 addresses (in a
// different ORDER only) and therefore compute the same product -- only
// the miss counts differ.
void matmul_miss_triple(int N, int tile1, int tile2,
                         long a_base, long b_base, long c_base, long* out);
