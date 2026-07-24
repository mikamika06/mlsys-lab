#include "sol.hpp"

long simulate_aos_invalidations(int num_threads, int num_increments) {
    reset_invalidations();
    for (int i = 0; i < num_increments; i++) {
        for (int t = 0; t < num_threads; t++) {
            long addr = (long)t * 4;
            report_write(t, addr);
        }
    }
    return total_invalidations();
}

long simulate_padded_invalidations(int num_threads, int num_increments) {
    reset_invalidations();
    for (int i = 0; i < num_increments; i++) {
        for (int t = 0; t < num_threads; t++) {
            long addr = (long)t * CACHE_LINE_BYTES;
            report_write(t, addr);
        }
    }
    return total_invalidations();
}
