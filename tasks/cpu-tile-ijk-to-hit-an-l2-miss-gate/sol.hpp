#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 32 sets, 4-way -- 8192 bytes
// total capacity). Real hardware cache timing isn't reproducible across
// machines, so this model -- not the CPU's actual cache -- is the sole
// source of every miss count the driver prints. Call touch() once per
// byte address you access.
void touch(long byte_addr);

// Row-major address helpers for three N x N matrices of 4-byte elements,
// laid out back-to-back: A first, then B, then C.
inline long a_addr(int N, int i, int k) { return (long)(i * N + k) * 4; }
inline long b_addr(int N, int k, int j) { return (long)N * N * 4 + (long)(k * N + j) * 4; }
inline long c_addr(int N, int i, int j) { return 2L * N * N * 4 + (long)(i * N + j) * 4; }

// C += A x B: for every (i, j, k) in [0,N) x [0,N) x [0,N), touch
// a_addr(N,i,k), b_addr(N,k,j), then c_addr(N,i,j), EXACTLY ONCE each --
// but process the N x N x N index space in T x T x T TILES: pick a tile
// of `i`, a tile of `j`, and a tile of `k` (looping over tiles in
// whichever tile order you like), and finish every (i, j, k) triple
// inside that tile before moving to the next one, instead of one single
// naive i-j-k sweep over the whole ranges. This keeps each tile's A/B/C
// sub-blocks cache-resident while they're being reused.
void tiled_matmul(int N, int T);
