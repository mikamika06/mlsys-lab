#include "sol.hpp"

// Fixed: A_colmajor's memory, read row-major, is the K x M transpose of A.
// TransA = CblasTrans undoes that; lda is the K x M view's row length (M).
void gemm_with_col_major_a(int M, int N, int K,
                            const double* A_colmajor, const double* B_rowmajor,
                            double* C_rowmajor) {
    row_major_gemm(CblasTrans, CblasNoTrans, M, N, K,
                    1.0, A_colmajor, /*lda=*/M,
                    B_rowmajor, /*ldb=*/N,
                    0.0, C_rowmajor, /*ldc=*/N);
}
