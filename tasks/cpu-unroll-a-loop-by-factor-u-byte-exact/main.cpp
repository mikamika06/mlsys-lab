// Fixed driver: one fixed (x, y_init, a) triple, run through
// axpy_unrolled at every divisor of n=12, printing the resulting y each
// time (reset from y_init before every trial). No timing, no
// randomness -- every input is a fixed formula.
#include "sol.hpp"
#include <cstdio>

namespace {
const int N = 12;
const float A = 1.7f;

void fill_inputs(float* x, float* y_init) {
    for (int i = 0; i < N; i++) {
        x[i] = 0.5f + 0.25f * static_cast<float>(i);
        y_init[i] = 1.0f - 0.1f * static_cast<float>(i);
    }
}
} // namespace

int main() {
    float x[N], y_init[N], y[N];
    fill_inputs(x, y_init);

    static const int U_VALUES[] = {1, 2, 3, 4, 6, 12};
    const int NUM_U = sizeof(U_VALUES) / sizeof(U_VALUES[0]);

    for (int t = 0; t < NUM_U; t++) {
        int U = U_VALUES[t];
        for (int i = 0; i < N; i++) y[i] = y_init[i];
        axpy_unrolled(N, U, A, x, y);

        printf("U=%d:", U);
        for (int i = 0; i < N; i++) printf(" %.6g", static_cast<double>(y[i]));
        printf("\n");
    }
    return 0;
}
