#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// This is the layout arithmetic a C++ extension must get right before it can
// hand off to a BLAS call like `cblas_sgemm(..., CblasTrans, CblasNoTrans,
// ...)`: mapping row-major, strided, (possibly transposed) flat buffers into
// the matrix product C = alpha * A^T * B + beta * C.
//
//   * `flatA` physically holds a K x M row-major matrix with row stride
//     `lda` (each physical row has `lda` elements, only the first M are
//     meaningful). Row k of that physical matrix starts at
//     `flatA[k * lda]`. A^T is therefore M x K with
//         A^T[m][k] = flatA[k * lda + m].
//   * `flatB` physically holds a K x N row-major matrix with row stride
//     `ldb`:  B[k][n] = flatB[k * ldb + n].
//   * `flatC` physically holds an M x N row-major matrix with row stride
//     `ldc`, both read (for the `beta * C` term) and written:
//         C[m][n] = flatC[m * ldc + n].
//
// Mutate `flatC` in place so that, for every m in [0,M) and n in [0,N),
//     flatC[m*ldc+n] = alpha * sum_k A^T[m][k] * B[k][n] + beta * flatC[m*ldc+n]
// using the OLD value of flatC[m*ldc+n] on the right-hand side. Any element
// of `flatC` outside that M x N logical window (i.e. row-stride padding
// beyond column N) must be left untouched.
// ---------------------------------------------------------------------------
void sgemm_at_b(int M, int N, int K, float alpha, float beta,
                 const std::vector<float>& flatA, int lda,
                 const std::vector<float>& flatB, int ldb,
                 std::vector<float>& flatC, int ldc);
