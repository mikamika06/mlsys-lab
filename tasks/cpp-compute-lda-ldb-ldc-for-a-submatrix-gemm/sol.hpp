#pragma once

// ============================================================================
// Fixed descriptor struct (real C++ — its size comes from the actual
// compiler, not a modeled ABI). 6 ints then 3 longs: the ints total 24
// bytes, already a multiple of the 8-byte alignment `long` needs, so no
// padding is inserted before them.
// ============================================================================
struct GemmDescriptor {
    int m, n, k;
    int lda, ldb, ldc;
    long offset_a, offset_b, offset_c;
};

// Row-major leading dimensions + element offsets for three submatrices A, B,
// C living inside one larger row-major buffer W that has W_cols columns.
struct GemmParams {
    int lda, ldb, ldc;
    long offset_a, offset_b, offset_c;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A (M x K), B (K x N) and C (M x N) are submatrices of a row-major buffer W
// with W_cols columns. A's top-left corner is at (a_row, a_col), B's at
// (b_row, b_col), C's at (c_row, c_col). Because all three live inside the
// SAME parent buffer, every leading dimension (the element stride between
// consecutive rows of a submatrix) is W_cols, regardless of the submatrix's
// own M/N/K shape — only the starting element offset differs.
//
// offset_x = x_row * W_cols + x_col  (row-major element index of the
// submatrix's top-left corner within the flat W buffer).
// ============================================================================
GemmParams submatrix_gemm_params(int M, int N, int K, int W_cols,
                                  int a_row, int a_col, int b_row, int b_col,
                                  int c_row, int c_col);
