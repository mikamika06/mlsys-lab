#include <cstdio>
#include "sol.hpp"

// FIXED driver. Dimensions are deliberately NOT multiples of block_size (10,
// 9, 7 against a tile of 4), so a blocking scheme that mishandles the
// leftover partial tile at the edge of a dimension shows up as wrong values
// in the last row/column, not just a global error.
int main() {
    const int M = 10, N = 9, K = 7, BS = 4;
    const float alpha = 1.5f, beta = 0.5f;

    static float A[M * K];
    static float B[K * N];
    static float C[M * N];

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A[i * K + k] = (float)((i * 7 + k * 3) % 11 - 5) * 0.3f;

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B[k * N + j] = (float)((k * 5 + j * 2) % 9 - 4) * 0.25f;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = (float)((i + j) % 7 - 3) * 0.5f;

    blocked_gemm(A, B, C, M, N, K, BS, alpha, beta);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%.4f ", C[i * N + j]);
        printf("\n");
    }
    return 0;
}
