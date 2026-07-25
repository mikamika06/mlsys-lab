#include "sol.hpp"

void compute_stats(const float* x, int n, float* out_sum, float* out_sumsq) {
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = 0; i < n; ++i) {
        touch_byte(reinterpret_cast<long>(&x[i]));
        float v = x[i];
        sum += v;
        sumsq += v * v;
    }
    *out_sum = sum;
    *out_sumsq = sumsq;
}
