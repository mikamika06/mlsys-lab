#include "sol.hpp"

void fit_roofline(const double* ai, const double* attained, int n, double* peak_bw_out, double* peak_flops_out) {
    int min_i = 0, max_i = 0;
    for (int i = 1; i < n; i++) {
        if (ai[i] < ai[min_i]) min_i = i;
        if (ai[i] > ai[max_i]) max_i = i;
    }
    *peak_bw_out = attained[min_i] / ai[min_i];
    *peak_flops_out = attained[max_i];
}
