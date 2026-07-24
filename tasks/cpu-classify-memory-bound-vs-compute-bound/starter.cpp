#include "sol.hpp"

// TODO: compute ridge = peak_flops / peak_bw, then for each i set
// out[i] = 1 if ai[i] >= ridge (compute-bound) else 0 (memory-bound).
void classify_regimes(double peak_flops, double peak_bw, const double* ai, int n, int* out) {
    (void)peak_flops; (void)peak_bw; (void)ai; (void)n; (void)out;
    // your code here
}
