#pragma once

// Deterministic direct-mapped cache model (harness code, defined in
// main.cpp): 16KB total, 64-byte lines. touch_byte(addr) simulates
// reading the 4-byte float at byte address `addr` through this cache and
// counts a MISS whenever that cache line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// Simulate the memory traffic of computing the FULL seq_len x seq_len
// attention score matrix S = Q @ K^T, by calling touch_byte() for every
// float element that would actually be read from memory, for every
// (i, j) score S[i][j].
//
// Q is always stored row-major [seq_len, head_dim]: Q[i]'s element d is
// at byte address q_base + (i*head_dim + d)*4 -- head_dim contiguous
// floats per row.
//
// K's layout depends on `layout`:
//   layout == 0 (row-major):   K[j]'s element d is at
//                              k_base + (j*head_dim + d)*4
//                              (contiguous, same shape as Q)
//   layout == 1 (transposed):  K[j]'s element d is at
//                              k_base + (d*seq_len + j)*4
//                              (STRIDED by seq_len*4 bytes across d --
//                              this is "K^T" stored [head_dim, seq_len])
//
// For every i in [0, seq_len) and every j in [0, seq_len), touch every
// element of Q[i] (head_dim touches) and every element of K[j] under the
// given layout (head_dim touches).
void simulate_score_matrix_traffic(int layout, int seq_len, int head_dim,
                                    long q_base, long k_base);

// Run simulate_score_matrix_traffic for BOTH layouts (resetting the cache
// between runs) at the given (seq_len, head_dim), using q_base=0 and
// k_base=1000000 for both, and return which layout produces FEWER cache
// misses: 0 for row-major K, 1 for the transposed layout.
int pick_better_layout(int seq_len, int head_dim);
