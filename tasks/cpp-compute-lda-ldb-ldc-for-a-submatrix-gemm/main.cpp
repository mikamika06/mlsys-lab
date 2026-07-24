#include <cstdio>
#include "sol.hpp"

// FIXED driver: builds a deterministic W_ROWS x W_COLS row-major buffer of
// doubles, asks the learner for lda/ldb/ldc + element offsets locating three
// submatrices A (M x K), B (K x N), C (M x N) inside it, performs the real
// GEMM C = A * B in place using ONLY those parameters (never the raw
// (row, col) pairs), and prints the parameters plus a checksum of the whole
// buffer afterward. Wrong parameters land the write in the wrong place (or
// read the wrong stride), which changes the checksum.

constexpr int W_ROWS = 20, W_COLS = 20;
constexpr int M = 3, N = 4, K = 5;
constexpr int A_ROW = 1, A_COL = 2;
constexpr int B_ROW = 5, B_COL = 6;
constexpr int C_ROW = 10, C_COL = 11;

int main() {
    double W[W_ROWS * W_COLS];
    for (int i = 0; i < W_ROWS * W_COLS; ++i) {
        W[i] = static_cast<double>((i * 37) % 97) / 10.0;
    }

    GemmParams p = submatrix_gemm_params(M, N, K, W_COLS, A_ROW, A_COL, B_ROW, B_COL, C_ROW, C_COL);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int l = 0; l < K; ++l) {
                acc += W[p.offset_a + i * p.lda + l] * W[p.offset_b + l * p.ldb + j];
            }
            W[p.offset_c + i * p.ldc + j] = acc;
        }
    }

    double checksum = 0.0;
    for (int i = 0; i < W_ROWS * W_COLS; ++i) checksum += W[i];

    printf("sizeof=%d\n", static_cast<int>(sizeof(GemmDescriptor)));
    printf("%d %d %d %ld %ld %ld\n", p.lda, p.ldb, p.ldc, p.offset_a, p.offset_b, p.offset_c);
    printf("checksum=%.6f\n", checksum);
    return 0;
}
