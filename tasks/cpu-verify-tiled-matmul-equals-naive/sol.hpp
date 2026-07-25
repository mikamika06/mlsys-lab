#pragma once

// Compute C = A * B for N x N row-major float matrices, using the
// standard cache-blocked (tiled) algorithm: loop over block indices
// (ii, jj, kk) in steps of `tile`, and within each block do the
// ordinary triple loop over the (possibly partial, at the N boundary)
// tile. C must be zero-initialized before accumulating; every C[i][j]
// must end up equal to sum_k A[i][k]*B[k][j], the same value the naive
// (unblocked) triple loop would produce -- blocking only changes the
// order additions happen in, never which numbers get multiplied and
// summed. `tile` need not evenly divide N.
void tiled_matmul(const float* A, const float* B, float* C, int N, int tile);
