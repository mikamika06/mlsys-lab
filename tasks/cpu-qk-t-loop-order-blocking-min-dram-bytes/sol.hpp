#pragma once

// Cache-access hook, DEFINED in main.cpp: a single-level set-associative
// LRU cache -- 64-byte lines, 64 sets, 8-way (32768 bytes total). Real
// hardware cache timing isn't reproducible across machines, so this
// model is the sole source of every miss count below. Call touch() once
// per BYTE ADDRESS you access, in the exact order your loop reads it.
void touch(long byte_addr);

// Emit the access pattern of a QK^T attention-score contraction: Q is
// S x d, K is S x d, both row-major and elem_bytes-wide, laid out
// back-to-back in one address space (Q first, K starting right after
// it). For every score[i][j] = sum_k Q[i,k]*K[j,k], touch Q[i,k] then
// K[j,k]:
//
//   touch((long)(i * d + k) * elem_bytes);                     // Q[i,k]
//   touch(baseK + (long)(j * d + k) * elem_bytes);             // K[j,k]
//   where baseK = (long)S * d * elem_bytes
//
// for every (i, j, k) in [0,S) x [0,S) x [0,d), exactly once each.
// Q and K here are each bigger than the whole cache, so streaming all of
// K once per row of Q (the naive i-j-k order) evicts and re-fetches K
// from scratch on almost every row. Tile the i and j loops in blocks of
// B instead, so a B x B block of Q-rows/K-rows stays cache-resident
// across the inner k loop for many (i,j) pairs at once, before moving on
// to the next tile -- this is exactly what a blocked/tiled matmul does
// to cut DRAM traffic.
void qkt_access(int S, int d, int B, int elem_bytes);
