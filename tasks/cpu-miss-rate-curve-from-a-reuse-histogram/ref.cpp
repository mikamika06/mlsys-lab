#include "sol.hpp"

double miss_rate_at_cache_size(const long* hist, int max_dist, long cold_misses,
                                long total_accesses, int cache_size) {
    long misses = cold_misses;
    for (int d = cache_size; d < max_dist; d++) {
        if (d >= 0) misses += hist[d];
    }
    return (double)misses / (double)total_accesses;
}
