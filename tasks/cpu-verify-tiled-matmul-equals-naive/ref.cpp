#include "sol.hpp"

void tiled_matmul(const float* A, const float* B, float* C, int N, int tile) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;

    for (int ii = 0; ii < N; ii += tile) {
        int i_end = ii + tile < N ? ii + tile : N;
        for (int jj = 0; jj < N; jj += tile) {
            int j_end = jj + tile < N ? jj + tile : N;
            for (int kk = 0; kk < N; kk += tile) {
                int k_end = kk + tile < N ? kk + tile : N;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        float acc = C[i * N + j];
                        for (int k = kk; k < k_end; k++) {
                            acc += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = acc;
                    }
                }
            }
        }
    }
}
