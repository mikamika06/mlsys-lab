#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache. Real hardware cache timing isn't
// reproducible across machines, so this model -- not the CPU's actual
// cache, and NOT derived from real pointer values (ASLR / allocator
// layout would make that nondeterministic across builds) -- is the sole
// source of every miss count the driver prints. Call touch() once per
// SYNTHETIC byte address, computed with in_addr()/out_addr() below, once
// per element you access -- never derive the address from an actual
// C++ pointer.
void touch(long byte_addr);

// Synthetic (not real-pointer-derived) row-major address helpers for two
// conceptual N x N matrices of 4-byte elements: `in` starts at byte 0,
// `out` starts immediately after it. Use these ONLY to compute what to
// pass to touch() -- the REAL data lives in the `in`/`out` float arrays
// passed to co_transpose below.
inline long in_addr(int N, int row, int col) { return (long)(row * N + col) * 4; }
inline long out_addr(int N, int row, int col) {
    return (long)N * N * 4 + (long)(row * N + col) * 4;
}

// Cache-oblivious recursive transpose, on REAL memory: out[col][row] =
// in[row][col] for the full N x N matrix (N a power of two, N > 8),
// computed without ever reading a cache parameter.
//
// Recursively split the FULL index space into quadrants: for every
// (row, col) pair with row in [r0, r0+n) and col in [c0, c0+n), once
// n <= 8 (the base case), for EACH such pair:
//   1. actually copy the data:  out[col*N + row] = in[row*N + col];
//   2. touch(in_addr(N,row,col)) then touch(out_addr(N,col,row))
//      -- EXACTLY once each, matching step 1's read/write.
// While n > 8, instead of the flat double loop above, recurse on the
// four (n/2) x (n/2) quadrants (r0,c0), (r0,c0+n/2), (r0+n/2,c0),
// (r0+n/2,c0+n/2), in any order. Write your own recursive helper (it
// needs r0, c0, n plus the fixed N, in, out) in solve.cpp; only this
// top-level signature is part of the contract main.cpp calls, and it
// always starts the recursion at r0=0, c0=0, n=N.
void co_transpose(const float* in, float* out, int N);
