#include "sol.hpp"

static long run_tile(int b, int passes) {
    reset_cache();
    for (int p = 0; p < passes; p++) {
        for (int f = 0; f < 3; f++) {
            for (int r = 0; r < b; r++) {
                for (int c = 0; c < b; c++) {
                    long addr = (long)(f * b * b + r * b + c) * 4;
                    touch_byte(addr);
                }
            }
        }
    }
    return miss_count();
}

int pick_resident_tile(int tile_b0, int tile_b1, int passes, long* out_misses) {
    out_misses[0] = run_tile(tile_b0, passes);
    out_misses[1] = run_tile(tile_b1, passes);
    return (out_misses[0] <= out_misses[1]) ? 0 : 1;
}
