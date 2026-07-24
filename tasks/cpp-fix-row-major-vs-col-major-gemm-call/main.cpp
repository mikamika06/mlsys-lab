#include <cstdio>
#include "sol.hpp"

void row_major_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int M, int N, int K,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                double a = (transA == CblasNoTrans) ? A[i * lda + k] : A[k * lda + i];
                double b = (transB == CblasNoTrans) ? B[k * ldb + j] : B[j * ldb + k];
                acc += a * b;
            }
            C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
        }
    }
}

static void run_case(int M, int N, int K) {
    // A_colmajor: M x K, column-major, A[i][j] = A_colmajor[i + j*M].
    double* A = new double[M * K];
    for (int j = 0; j < K; j++)
        for (int i = 0; i < M; i++)
            A[i + j * M] = (double)(i * 3 - j * 2 + 1);   // deterministic pattern

    // B_rowmajor: K x N, row-major.
    double* B = new double[K * N];
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B[k * N + j] = (double)(k - j + 5) * 0.5;

    // C_rowmajor: M x N, row-major, zeroed.
    double* C = new double[M * N];
    for (int i = 0; i < M * N; i++) C[i] = 0.0;

    gemm_with_col_major_a(M, N, K, A, B, C);

    printf("M=%d N=%d K=%d C=", M, N, K);
    for (int i = 0; i < M * N; i++) printf(" %.4f", C[i]);
    printf("\n");

    delete[] A; delete[] B; delete[] C;
}

int main() {
    run_case(3, 4, 5);
    run_case(2, 3, 4);
    run_case(4, 2, 3);
    return 0;
}
