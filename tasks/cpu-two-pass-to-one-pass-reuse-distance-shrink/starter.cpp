#include "sol.hpp"

// VALID but suboptimal: computes the same two statistics correctly, but
// reads x twice -- once per statistic -- instead of fusing them into one
// pass.
void compute_stats(const float* x, int n, float* out_sum, float* out_sumsq) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        touch_byte(reinterpret_cast<long>(&x[i]));
        sum += x[i];
    }
    float sumsq = 0.0f;
    for (int i = 0; i < n; ++i) {
        touch_byte(reinterpret_cast<long>(&x[i]));
        sumsq += x[i] * x[i];
    }
    *out_sum = sum;
    *out_sumsq = sumsq;
}
