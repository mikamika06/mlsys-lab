#include "sol.hpp"

// BUG: treats A_colmajor as if it were already a row-major M x K matrix
// (TransA = CblasNoTrans, lda = K). That is wrong for column-major memory
// and produces a garbled/transposed result.
void gemm_with_col_major_a(int M, int N, int K,
                            const double* A_colmajor, const double* B_rowmajor,
                            double* C_rowmajor) {
    row_major_gemm(CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0, A_colmajor, /*lda=*/K,
                    B_rowmajor, /*ldb=*/N,
                    0.0, C_rowmajor, /*ldc=*/N);
}
