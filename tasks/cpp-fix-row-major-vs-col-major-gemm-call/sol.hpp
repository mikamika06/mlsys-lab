#pragma once

// CBLAS-style enums, matching real cblas.h numeric values.
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };

// A real row-major-only GEMM kernel (harness code, defined in main.cpp),
// matching the calling convention of cblas_dgemm(order=CblasRowMajor, ...):
//
//     C := alpha * op(A) * op(B) + beta * C
//
// where op(X) = X if trans==CblasNoTrans, else X^T. `lda`/`ldb`/`ldc` are
// the classic BLAS leading dimensions of the AS-STORED (pre-transpose)
// row-major matrices — e.g. if transA==CblasTrans, A is stored as a K x M
// row-major matrix (op(A) = A^T is M x K) and `lda` is that stored
// matrix's row length.
void row_major_gemm(CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                     int M, int N, int K,
                     double alpha, const double* A, int lda,
                     const double* B, int ldb,
                     double beta, double* C, int ldc);

// A_colmajor is a COLUMN-MAJOR M x K matrix: A_colmajor[i + j*M] holds
// logical element (row i, col j). B_rowmajor is a plain ROW-MAJOR K x N
// matrix (ldb = N) and C_rowmajor is a plain ROW-MAJOR M x N output
// buffer (ldc = N), already zeroed by the caller.
//
// Compute C_rowmajor := A @ B (logical M x K times K x N) by calling
// row_major_gemm with the transpose flag and leading dimension that make
// it read A_colmajor's memory correctly WITHOUT copying or manually
// transposing it.
//
// Key fact: a column-major M x K matrix has the exact same byte layout as
// a row-major K x M matrix (A^T stored row-major). So pass A_colmajor to
// row_major_gemm as if it were that K x M row-major matrix — with
// TransA = CblasTrans and lda = M (the row length of the K x M view) —
// and let the kernel's own transpose undo it back to M x K.
void gemm_with_col_major_a(int M, int N, int K,
                            const double* A_colmajor, const double* B_rowmajor,
                            double* C_rowmajor);
