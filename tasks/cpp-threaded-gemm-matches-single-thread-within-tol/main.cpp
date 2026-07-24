#include <cstdio>
#include "sol.hpp"

// FIXED driver. Builds deterministic integer-valued inputs (exact in float,
// so a correct GEMM is bit-identical regardless of accumulation order or
// thread count), runs gemm() at several thread counts, and prints C plus a
// checksum for each run. A correct threaded GEMM prints the same numbers for
// every thread count; the single-thread reference prints the baseline.
int main() {
    const int M = 16, N = 16, K = 24;
    static float A[M * K], B[K * N], C[M * N];

    // Deterministic inputs in small integer ranges: products/sums stay well
    // below 2^24, so float accumulation is exact -> tolerance-robust.
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[i * K + k] = (float)(((i * K + k) % 5) - 2);   // {-2..2}
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B[k * N + j] = (float)(((k * N + j) % 7) - 3);   // {-3..3}

    const int thread_counts[3] = {1, 2, 4};
    for (int t = 0; t < 3; ++t) {
        for (int idx = 0; idx < M * N; ++idx) C[idx] = 0.0f;
        gemm(A, B, C, M, N, K, thread_counts[t]);

        for (int idx = 0; idx < M * N; ++idx) printf("%.6f ", C[idx]);
        double checksum = 0.0;
        for (int idx = 0; idx < M * N; ++idx) checksum += C[idx];
        printf("\nthreads=%d checksum=%.6f\n", thread_counts[t], checksum);
    }
    return 0;
}
