#pragma once

// Harness hook (declared here, DEFINED in main.cpp): a deterministic
// 4-way, 32-set, 64-byte-line LRU cache (8192 bytes total). touch(addr)
// simulates reading/writing the 4-byte float at simulated byte address
// `addr` through this cache and counts a MISS whenever that cache line
// wasn't already resident.
void reset_cache();
void touch(long byte_addr);
long miss_count();

// Row-major address helper: element (r, c) of a matrix whose row length
// is `stride`, starting at simulated base address `base`, lives at
// simulated address `base + (r*stride + c)*4`.
inline long mat_addr(long base, int stride, int r, int c) {
    return base + (long)(r * stride + c) * 4;
}

// Einsum contraction Y[b][i] = sum_j X[b][j] * W[i][j], for every
// b in [0, B), i in [0, I) -- a linear-layer-style contraction over the
// shared axis j. X is a real B x J row-major array, W a real I x J
// row-major array, Y a real B x I row-major array (use them for the
// actual float arithmetic); x_base/w_base/y_base are SIMULATED byte
// addresses for the cache trace only (independent of the real
// pointers -- use mat_addr(x_base, J, b, j) etc.).
//
// Requirement: for every (b, i, j) triple, touch mat_addr(x_base,J,b,j),
// then mat_addr(w_base,J,i,j), then mat_addr(y_base,I,b,i) -- exactly
// once each -- and accumulate Y[b*I+i] += X[b*J+j] * W[i*J+j]. Returns
// the sum of every element of Y once the full contraction is done (a
// checksum the driver uses to confirm the math is unchanged).
float einsum_bij(int B, int I, int J,
                  long x_base, long w_base, long y_base,
                  const float* X, const float* W, float* Y);
