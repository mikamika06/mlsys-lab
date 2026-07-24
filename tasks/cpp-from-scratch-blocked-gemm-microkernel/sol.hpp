#pragma once

// Cache-blocked GEMM: compute C := alpha * (A @ B) + beta * C in place, with
// tile size `block_size`. A is M x K, B is K x N, C is M x N, all row-major
// float32, contiguous (A[i*K+k], B[k*N+j], C[i*N+j]). M, N, K need not be
// multiples of block_size -- the last tile along each dimension may be
// smaller.
void blocked_gemm(const float* A, const float* B, float* C,
                   int M, int N, int K, int block_size,
                   float alpha, float beta);
