#include "sol.hpp"

// BUG: three separate full scans over the records -- one for sum, one
// for min, one for max -- each re-touching the same widely-strided hot
// field from scratch. The sum/min/max values come out correct, but the
// working set is far bigger than the cache, so none of the 3 passes can
// reuse another's cache lines: 3x the necessary misses. Fix: compute all
// 3 reductions in ONE pass, touching each record's hot field once.
void hot_field_stats(long base, int stride, int hot_offset, int n, double* out) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        long addr = base + (long)i * stride + hot_offset;
        touch_byte(addr);
        double v = (double)((i * 37) % 97) - 48.0;
        sum += v;
    }

    double mn = 1e18;
    for (int i = 0; i < n; i++) {
        long addr = base + (long)i * stride + hot_offset;
        touch_byte(addr);
        double v = (double)((i * 37) % 97) - 48.0;
        if (v < mn) mn = v;
    }

    double mx = -1e18;
    for (int i = 0; i < n; i++) {
        long addr = base + (long)i * stride + hot_offset;
        touch_byte(addr);
        double v = (double)((i * 37) % 97) - 48.0;
        if (v > mx) mx = v;
    }

    out[0] = sum;
    out[1] = mn;
    out[2] = mx;
}
