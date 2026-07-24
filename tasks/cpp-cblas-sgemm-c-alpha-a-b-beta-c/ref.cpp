#include "sol.hpp"

void sgemm_at_b(int M, int N, int K, float alpha, float beta,
                 const std::vector<float>& flatA, int lda,
                 const std::vector<float>& flatB, int ldb,
                 std::vector<float>& flatC, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float at_mk = flatA[k * lda + m];   // A^T[m][k]
                float b_kn  = flatB[k * ldb + n];   // B[k][n]
                sum += at_mk * b_kn;
            }
            int idx = m * ldc + n;
            flatC[idx] = alpha * sum + beta * flatC[idx];
        }
    }
}
