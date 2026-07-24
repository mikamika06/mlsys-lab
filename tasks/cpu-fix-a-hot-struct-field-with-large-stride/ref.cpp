#include "sol.hpp"

// FIXED: a single fused pass computes sum, min, and max together, so
// each record's hot field is touched exactly once.
void hot_field_stats(long base, int stride, int hot_offset, int n, double* out) {
    double sum = 0.0, mn = 1e18, mx = -1e18;
    for (int i = 0; i < n; i++) {
        long addr = base + (long)i * stride + hot_offset;
        touch_byte(addr);
        double v = (double)((i * 37) % 97) - 48.0;
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    out[0] = sum;
    out[1] = mn;
    out[2] = mx;
}
