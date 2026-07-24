#include <cstdio>
#include <vector>
#include "sol.hpp"

// PROVIDED. Deterministic value generator (no rand(), no clock): a small
// integer hash mapped into [-1, 1]. Same sequence every run, every platform.
static float detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 12345u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return ((float)(x % 2000) / 1000.0f) - 1.0f;
}

// FIXED driver. Do not edit. M x N x K product with row strides strictly
// larger than the logical dimensions (so stride handling actually matters),
// deterministic A/B/C contents, calls the learner's sgemm_at_b, then prints
// every element of the full physical flatC buffer (including stride
// padding) so both correctness and "don't touch what you shouldn't" are
// visible in the output.
int main() {
    const int M = 4, N = 5, K = 3;
    const int lda = K + 2;   // physical row stride of A (K x M matrix)
    const int ldb = N + 2;   // physical row stride of B (K x N matrix)
    const int ldc = N + 3;   // physical row stride of C (M x N matrix)
    const float alpha = 1.5f, beta = 0.5f;

    std::vector<float> flatA(K * lda), flatB(K * ldb), flatC(M * ldc);
    for (int i = 0; i < (int)flatA.size(); ++i) flatA[i] = detval(i);
    for (int i = 0; i < (int)flatB.size(); ++i) flatB[i] = detval(i + 1000);
    for (int i = 0; i < (int)flatC.size(); ++i) flatC[i] = detval(i + 2000);

    sgemm_at_b(M, N, K, alpha, beta, flatA, lda, flatB, ldb, flatC, ldc);

    for (int i = 0; i < (int)flatC.size(); ++i) printf("%.6f ", flatC[i]);
    printf("\n");
    return 0;
}
