#include "sol.hpp"

long simulate_decode_pass(int T, int rec_bytes, int prefetch_distance) {
    cache_reset();
    long misses = 0;
    for (int t = 0; t < T; t++) {
        if (!touch((long)t * rec_bytes)) misses++;
        if (prefetch_distance > 0 && t + prefetch_distance < T) {
            touch((long)(t + prefetch_distance) * rec_bytes);
        }
    }
    return misses;
}

int choose_best_prefetch_distance(int T, int rec_bytes, int max_distance) {
    int best_d = 0;
    long best_misses = simulate_decode_pass(T, rec_bytes, 0);
    for (int d = 1; d <= max_distance; d++) {
        long m = simulate_decode_pass(T, rec_bytes, d);
        if (m < best_misses) {
            best_misses = m;
            best_d = d;
        }
    }
    return best_d;
}
