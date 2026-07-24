#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Write out = transpose(in) for an n x n row-major matrix of doubles
// (out[j*n+i] = in[i*n+j] for every i, j), but do it BLOCKED: process the
// output in `block x block` tiles (loop over tile row/column, then over
// the elements inside each tile), so a whole tile's worth of both the
// source and destination stays in a small working set instead of striding
// across the full matrix on every element. Blocking changes the ORDER
// elements are visited in, never their VALUES — the result must be
// byte-for-byte identical to the naive, unblocked transpose.
//
// n is always an exact multiple of block.
// ============================================================================
void blocked_transpose(const double* in, double* out, int n, int block);
