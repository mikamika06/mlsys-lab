#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 32 sets, 4-way -- 8192 bytes
// total capacity). Real hardware cache timing isn't reproducible across
// machines, so this model -- not the CPU's actual cache -- is the sole
// source of every miss count the driver prints. Call touch() once per
// byte address you access.
void touch(long byte_addr);

// Row-major address helpers for two N x N matrices of 4-byte elements:
// `in` starts at byte 0, `out` starts immediately after it.
inline long in_addr(int N, int r, int c) { return (long)(r * N + c) * 4; }
inline long out_addr(int N, int r, int c) {
    return (long)N * N * 4 + (long)(r * N + c) * 4;
}

// Transpose: out[j][i] = in[i][j]. For every (i, j) in [0,N) x [0,N),
// touch in_addr(N,i,j) then out_addr(N,j,i), EXACTLY ONCE each -- but
// process the N x N index space in B x B TILES: finish every (i, j) pair
// inside a tile before moving to the next tile, instead of sweeping the
// whole matrix row by row. This keeps each tile's `in`/`out` regions
// cache-resident while they're being reused.
void blocked_transpose(int N, int B);
