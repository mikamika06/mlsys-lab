#include <cstdio>
#include <cstdint>
#include "sol.hpp"

// FIXED driver: build deterministic integer-valued data (so the reduction is
// exact under any accumulation order), run the K-accumulator reduction, and
// print the K partial sums followed by the grand total.
int main() {
    const int N = 4096;
    const int K = 4;
    static double x[N];
    for (int i = 0; i < N; i++) {
        uint32_t h = (uint32_t)i * 1103515245u + 12345u;   // deterministic LCG step
        x[i] = (double)((int)((h >> 8) & 1023u) - 512);     // integer in [-512, 511]
    }

    double partial[K] = {};                 // zero-initialised; a wrong solve leaves them 0
    double total = reduce_multi_acc(x, N, partial, K);

    for (int j = 0; j < K; j++) printf("%.6f\n", partial[j]);
    printf("%.6f\n", total);
    return 0;
}
