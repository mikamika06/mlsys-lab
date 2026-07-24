#include "sol.hpp"

// TODO: find the smallest-ai sample (memory-bound: slope = attained/ai)
// and the largest-ai sample (compute-bound plateau: peak_flops =
// attained). See sol.hpp.
void fit_roofline(const double* ai, const double* attained, int n, double* peak_bw_out, double* peak_flops_out) {
    (void)ai; (void)attained; (void)n;
    *peak_bw_out = 0.0;
    *peak_flops_out = 0.0;
    // your code here
}
