#include "sol.hpp"

GemmParams submatrix_gemm_params(int M, int N, int K, int W_cols,
                                  int a_row, int a_col, int b_row, int b_col,
                                  int c_row, int c_col) {
    (void)M;
    (void)N;
    (void)K;
    GemmParams p;
    p.lda = W_cols;
    p.ldb = W_cols;
    p.ldc = W_cols;
    p.offset_a = static_cast<long>(a_row) * W_cols + a_col;
    p.offset_b = static_cast<long>(b_row) * W_cols + b_col;
    p.offset_c = static_cast<long>(c_row) * W_cols + c_col;
    return p;
}
