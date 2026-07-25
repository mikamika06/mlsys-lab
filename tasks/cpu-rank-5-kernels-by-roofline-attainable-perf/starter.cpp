#include "sol.hpp"

// TODO: for each kernel i, compute ai = flops[i]/bytes[i], attainable =
// min(peak_flops_per_sec, ai*peak_bytes_per_sec); sort the indices
// [0,n) by attainable DESCENDING into rank_out. See sol.hpp.
void rank_kernels_by_attainable_perf(const double* flops, const double* bytes, int n,
                                      double peak_flops_per_sec, double peak_bytes_per_sec,
                                      int* rank_out) {
    (void)flops; (void)bytes; (void)peak_flops_per_sec; (void)peak_bytes_per_sec;
    // your code here
    for (int i = 0; i < n; i++) rank_out[i] = i;
}
