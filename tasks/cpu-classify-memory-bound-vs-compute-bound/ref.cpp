#include "sol.hpp"

void classify_regimes(double peak_flops, double peak_bw, const double* ai, int n, int* out) {
    double ridge = peak_flops / peak_bw;
    for (int i = 0; i < n; i++) {
        out[i] = (ai[i] >= ridge) ? 1 : 0;
    }
}
