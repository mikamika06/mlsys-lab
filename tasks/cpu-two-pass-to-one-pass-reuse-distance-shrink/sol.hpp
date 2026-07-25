#pragma once

// ============================================================================
// Deterministic fully-associative LRU cache model (FIXED — defined in
// main.cpp): 64 lines x 64 bytes/line = 4096-byte cache. touch_byte(addr)
// simulates one memory access to byte address `addr`; it maps to line
// addr/64 and counts a MISS whenever that line was not already resident
// (bringing it in evicts the least-recently-used line if the cache is
// full).
// ============================================================================
void reset_cache();
void touch_byte(long addr);
long miss_count();

// ============================================================================
// Compute BOTH the sum and the sum-of-squares of x[0..n) -- the classic
// inputs to a mean/variance computation. n is large enough that the array
// does not fit in the modeled 4096-byte cache, so how many times you read
// each element back from memory matters: call touch_byte(&x[i]) exactly
// ONCE per element, in a SINGLE forward pass over x, computing both
// running sums together, rather than one pass per statistic.
// ============================================================================
void compute_stats(const float* x, int n, float* out_sum, float* out_sumsq);
