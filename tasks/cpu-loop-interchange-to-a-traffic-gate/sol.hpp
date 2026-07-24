#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache (64-byte lines, 32 sets, 4-way -- 8192 bytes
// total capacity). Real hardware cache timing isn't reproducible across
// machines, so this model -- not the CPU's actual cache -- is the sole
// source of every miss count the driver prints. Call touch() once per
// byte address you access.
void touch(long byte_addr);

// Row-major address helper for an N x N matrix of 8-byte (double)
// elements: element (row, col) lives at byte address
// (row * N + col) * 8.
inline long elem_addr(int N, int row, int col) {
    return (long)(row * N + col) * 8;
}

// Touch every element of the N x N matrix EXACTLY ONCE, in ROW-major
// order: for each row, walk every column of that row before moving to
// the next row (loop order `for (row) for (col)`, innermost loop
// stride-1 over `col`).
//
// This is the loop-interchanged version of the column-major traversal
// the harness uses as its "naive" baseline (`for (col) for (row)`, same
// n*n touches, same elements, only the visiting ORDER differs). Row-
// major order keeps consecutive touches inside the same 64-byte line
// (8 doubles per line) instead of jumping N*8 bytes apart on every
// single step.
void row_major_traverse(int N);
