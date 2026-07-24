#include "sol.hpp"

void blocked_gemm(const float* A, const float* B, float* C,
                   int M, int N, int K, int block_size,
                   float alpha, float beta) {
    // Scale C by beta up front, then accumulate alpha * A @ B into it.
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] *= beta;

    for (int ii = 0; ii < M; ii += block_size) {
        int i_max = (ii + block_size < M) ? ii + block_size : M;
        for (int jj = 0; jj < N; jj += block_size) {
            int j_max = (jj + block_size < N) ? jj + block_size : N;
            for (int kk = 0; kk < K; kk += block_size) {
                int k_max = (kk + block_size < K) ? kk + block_size : K;
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                        float a = alpha * A[i * K + k];
                        for (int j = jj; j < j_max; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
