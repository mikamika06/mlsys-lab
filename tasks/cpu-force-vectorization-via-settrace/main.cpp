#include <cstdio>
#include "sol.hpp"

int g_vector_ops = 0;
void op_tick() { g_vector_ops++; }

// FIXED driver: deterministic inputs, no timing, no rand().
int main() {
    const int n = 16;
    const int width = 4;  // one 128-bit NEON float32x4 register

    float a[n], b[n], c[n], out[n];
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1) * 0.5f;
        b[i] = (float)(n - i) * 0.25f;
        c[i] = (float)(i % 5) - 2.0f;
        out[i] = 0.0f;
    }

    fma_vectorized(a, b, c, out, n, width);

    double checksum = 0.0;
    for (int i = 0; i < n; i++) checksum += (double)out[i];

    printf("checksum=%.6f\n", checksum);
    printf("vector_ops=%d\n", g_vector_ops);
    return 0;
}
