#include "sol.hpp"

static long run_variant(long base, long stride, int count, int passes, long page_size) {
    reset_tlb(page_size);
    for (int p = 0; p < passes; p++) {
        for (int i = 0; i < count; i++) {
            touch(base + (long)i * stride);
        }
    }
    return miss_count();
}

void tlb_miss_pair(long base, long stride, int count, int passes, long* out) {
    out[0] = run_variant(base, stride, count, passes, 4096);
    out[1] = run_variant(base, stride, count, passes, 2L * 1024 * 1024);
}
