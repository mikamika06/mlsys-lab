#include "sol.hpp"

void generate_and_run(long base, int stride_bytes, int n_steps, long* out) {
    for (int k = 0; k < n_steps; k++) {
        long addr = base + (long)k * stride_bytes;
        touch_no_prefetch(addr);
        touch_next_line(addr);
        touch_stride(addr);
    }
    out[0] = miss_count_no_prefetch();
    out[1] = miss_count_next_line();
    out[2] = miss_count_stride();
}
