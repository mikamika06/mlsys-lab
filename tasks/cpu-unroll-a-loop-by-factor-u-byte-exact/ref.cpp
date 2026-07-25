#include "sol.hpp"

void axpy_unrolled(int n, int U, float a, const float* x, float* y) {
    for (int b = 0; b < n; b += U) {
        for (int k = 0; k < U; k++) {
            int i = b + k;
            y[i] = y[i] + a * x[i];
        }
    }
}
