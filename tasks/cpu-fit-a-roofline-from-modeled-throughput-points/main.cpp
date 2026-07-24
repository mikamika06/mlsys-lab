#include <cstdio>
#include <algorithm>
#include "sol.hpp"

// FIXED driver. True device: peak_flops = 20e12 FLOP/s, peak_bw = 2.5e12
// bytes/s -> ridge point = 8.0 FLOP/byte. 6 arithmetic-intensity samples
// straddle the ridge; attained throughput at each is computed from the
// roofline formula itself (not hardcoded), so it is real modeled data.
int main() {
    const double peak_flops_true = 20e12;
    const double peak_bw_true = 2.5e12;
    const int n = 6;
    double ai[n] = {0.5, 1.0, 2.0, 8.0, 20.0, 50.0};
    double attained[n];
    for (int i = 0; i < n; i++) {
        attained[i] = std::min(peak_flops_true, ai[i] * peak_bw_true);
    }

    double fitted_bw = -1.0, fitted_flops = -1.0;  // sentinel: an empty starter leaves these untouched
    fit_roofline(ai, attained, n, &fitted_bw, &fitted_flops);

    printf("fitted_peak_bw=%.6e fitted_peak_flops=%.6e\n", fitted_bw, fitted_flops);
    return 0;
}
