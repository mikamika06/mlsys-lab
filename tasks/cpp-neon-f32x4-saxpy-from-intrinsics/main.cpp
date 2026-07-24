#include <cstdio>
#include "sol.hpp"

// FIXED driver. Deterministic input, no randomness.
int main() {
    const int N = 16;              // multiple of 4 -> four full f32x4 blocks
    const float a = 2.5f;
    float x[N], y[N];
    for (int i = 0; i < N; i++) {
        x[i] = 0.5f * i - 3.0f;    // spans negative, zero, positive
        y[i] = 0.25f * i + 1.0f;
    }

    saxpy_neon(a, x, y, N);        // y <- a*x + y, in place

    double s = 0;
    for (int i = 0; i < N; i++) { printf("%.6f ", y[i]); s += y[i]; }
    printf("\nsum=%.6f\n", s);
    return 0;
}
