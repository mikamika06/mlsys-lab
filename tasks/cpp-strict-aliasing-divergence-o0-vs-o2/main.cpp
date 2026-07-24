#include <cstdio>
#include "sol.hpp"

// FIXED driver. Deterministic input, no randomness.
int main() {
    const int N = 8;
    const int KEEP = 10;                 // keep the top 10 of 23 mantissa bits
    float x[N] = { 3.141592f, 2.718281f, 1.414213f, 0.577215f,
                   -1.732050f, 123.456f, -0.001234f, 65.4321f };

    // Count the bits about to be discarded (on the original values), then quantize.
    int lost = count_bits_lost(x, N, KEEP);
    quantize_mantissa(x, N, KEEP);

    double s = 0;
    for (int i = 0; i < N; i++) { printf("%.6f ", x[i]); s += x[i]; }
    printf("\nsum=%.6f\n", s);
    printf("lost=%d\n", lost);
    return 0;
}
