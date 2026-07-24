#pragma once

// Cache access hook, DEFINED in main.cpp: a deterministic set-associative
// LRU cache. Real hardware cache timing isn't reproducible across
// machines, so this model -- not the CPU's actual cache -- is the sole
// source of every miss count the driver prints. The driver reconfigures
// this model's total CAPACITY between the 4 scenarios it runs (line size
// and associativity stay fixed); call touch() once per byte address you
// access and it always reflects whichever capacity is currently active.
void touch(long byte_addr);

// Row-major address helpers for two N x N matrices of 4-byte elements:
// `in` starts at byte 0, `out` starts immediately after it. N is always
// the FULL matrix side length -- even when addressing a sub-block, pass
// the global (row, col) pair, never a block-local one.
inline long in_addr(int N, int row, int col) { return (long)(row * N + col) * 4; }
inline long out_addr(int N, int row, int col) {
    return (long)N * N * 4 + (long)(row * N + col) * 4;
}

// Cache-oblivious recursive transpose: out[j][i] = in[i][j] for the full
// N x N matrix (N a power of two, N > 8), computed WITHOUT ever reading
// the cache's line size, capacity or associativity -- the exact same code
// path below must produce a low miss count at every one of the 4 cache
// sizes the driver tries it against, with no per-size retuning.
//
// Recursively split the FULL index space into quadrants: touch every
// (row, col) pair with row in [r0, r0+n) and col in [c0, c0+n) --
// in_addr(N,row,col) then out_addr(N,col,row), EXACTLY ONCE each -- but
// once n > 8, instead of a flat double loop over the whole n x n block,
// recurse on its four (n/2) x (n/2) quadrants (r0,c0), (r0,c0+n/2),
// (r0+n/2,c0), (r0+n/2,c0+n/2), in any order. When n <= 8, touch every
// pair in the block directly (the base case). Write your own recursive
// helper (it needs r0, c0, n in addition to the fixed N) in solve.cpp;
// only this top-level signature is part of the contract main.cpp calls,
// and it always starts the recursion at r0=0, c0=0, n=N.
void co_transpose(int N);
