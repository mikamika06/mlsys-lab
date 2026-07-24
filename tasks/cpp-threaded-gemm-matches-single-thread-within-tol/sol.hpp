#pragma once
// Row-major GEMM:  C = A * B
//   A is M x K, B is K x N, C is M x N.
//   Index layout: A[i*K + k], B[k*N + j], C[i*N + j].
//
// You may parallelize the M-row loop over up to `num_threads` std::thread
// workers (split the rows into contiguous blocks). Because each output
// element C[i*N+j] is the sum over k of one row of A and one column of B,
// row blocking keeps every element inside a single thread — so the result
// MUST be identical for any num_threads >= 1 (no data races, deterministic).
//
// C is expected to be preinitialized to 0 by the caller.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, int num_threads);
