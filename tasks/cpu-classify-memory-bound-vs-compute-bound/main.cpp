#include <cstdio>
#include "sol.hpp"

// FIXED driver. Device with peak_flops = 16e12 FLOP/s, peak_bw = 2e12
// bytes/s -> ridge point = 8.0 FLOP/byte exactly. AI values chosen to
// straddle the ridge point, including the exact boundary (which counts
// as compute-bound).
int main() {
    const double peak_flops = 16e12;
    const double peak_bw = 2e12;
    const int n = 8;
    double ai[n] = {1.0, 4.0, 7.999, 8.0, 8.001, 16.0, 100.0, 0.25};

    int out[n];
    for (int i = 0; i < n; i++) out[i] = -1;  // sentinel: an empty starter leaves this untouched

    classify_regimes(peak_flops, peak_bw, ai, n, out);

    printf("ridge=%.4f\n", peak_flops / peak_bw);
    for (int i = 0; i < n; i++) printf("%d ", out[i]);
    printf("\n");
    return 0;
}
