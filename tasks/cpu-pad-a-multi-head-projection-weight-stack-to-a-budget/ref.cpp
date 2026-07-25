#include "sol.hpp"
#include <climits>

int choose_padding_bytes(int H, int row_bytes, int line_bytes, int sets,
                          int ways, int max_pad_bytes) {
    int best_pad = 0;
    long best_misses = LONG_MAX;

    for (int pad = 0; pad <= max_pad_bytes; pad += 4) {
        reset_cache(line_bytes, sets, ways);
        long stride = (long)row_bytes + pad;
        for (int h = 0; h < H; h++) touch((long)h * stride);
        for (int h = 0; h < H; h++) touch((long)h * stride);
        long candidate_misses = miss_count();
        if (candidate_misses < best_misses) {
            best_misses = candidate_misses;
            best_pad = pad;
        }
    }
    return best_pad;
}
