#include "sol.hpp"

// Correct single-thread reference GEMM. This is the baseline the learner's
// (optionally threaded) implementation must match within tolerance. The
// result is independent of num_threads, so the reference simply ignores it.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, int num_threads) {
    (void)num_threads;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}
