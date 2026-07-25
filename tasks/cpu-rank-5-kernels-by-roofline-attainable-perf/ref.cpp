#include "sol.hpp"
#include <algorithm>
#include <numeric>

void rank_kernels_by_attainable_perf(const double* flops, const double* bytes, int n,
                                      double peak_flops_per_sec, double peak_bytes_per_sec,
                                      int* rank_out) {
    for (int i = 0; i < n; i++) rank_out[i] = i;

    std::stable_sort(rank_out, rank_out + n, [&](int a, int b) {
        double ai_a = flops[a] / bytes[a];
        double ai_b = flops[b] / bytes[b];
        double att_a = std::min(peak_flops_per_sec, ai_a * peak_bytes_per_sec);
        double att_b = std::min(peak_flops_per_sec, ai_b * peak_bytes_per_sec);
        return att_a > att_b;
    });
}
