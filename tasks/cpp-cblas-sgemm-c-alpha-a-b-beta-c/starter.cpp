#include "sol.hpp"

// TODO: compute flatC[m*ldc+n] = alpha * sum_k A^T[m][k] * B[k][n] +
// beta * flatC[m*ldc+n] for every m in [0,M), n in [0,N), where
// A^T[m][k] = flatA[k*lda+m] and B[k][n] = flatB[k*ldb+n]. See sol.hpp.
void sgemm_at_b(int M, int N, int K, float alpha, float beta,
                 const std::vector<float>& flatA, int lda,
                 const std::vector<float>& flatB, int ldb,
                 std::vector<float>& flatC, int ldc) {
    (void)M; (void)N; (void)K; (void)alpha; (void)beta;
    (void)flatA; (void)lda; (void)flatB; (void)ldb; (void)flatC; (void)ldc;
}
