#pragma once

// Deterministic set-associative LRU cache model (harness code, defined
// in main.cpp): 64-byte lines, 32 sets, 4-way (8192 bytes total).
// touch_byte(addr) simulates reading/writing the 4-byte float at byte
// address `addr` through this cache and counts a MISS whenever that
// cache line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// N x N row-major matrices A, B, C stored starting at a_base/b_base/
// c_base (element (r, c) of an N x N row-major matrix at `base` lives at
// byte address base + (r*N + c)*4). N is always a power of two.

// Naive matmul C = A*B: the textbook i-j-k triple loop over the WHOLE
// matrices, no blocking. For every i, j, k in [0, N), touch_byte() the
// address of A[i][k], B[k][j], and C[i][j] (in that order).
void naive_matmul(int N, long a_base, long b_base, long c_base);

// Cache-oblivious recursive matmul C = A*B: split each N x N matrix into
// four (N/2) x (N/2) quadrants once N > 8, and recurse on the 8 quadrant
// products that contribute to the 4 output quadrants:
//   C11 += A11*B11 + A12*B21      C12 += A11*B12 + A12*B22
//   C21 += A21*B11 + A22*B21      C22 += A21*B12 + A22*B22
// Each quadrant is a VIEW into the original matrix, not a copy: track its
// top-left byte address plus the ORIGINAL matrix's row stride N (in
// elements) so quadrant element (r, c) is at
// quadrant_base + (r*N + c)*4. When N <= 8, fall back to the same
// triple-loop touch pattern as naive_matmul (over just that N x N
// sub-block, using the full-matrix stride for addressing).
//
// You will need your own recursive helper (with stride parameters) --
// declare and define it yourself in solve.cpp; only this top-level
// signature is part of the contract main.cpp calls.
void recursive_matmul(int N, long a_base, long b_base, long c_base);
