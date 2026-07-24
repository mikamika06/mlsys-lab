#include "sol.hpp"

// TODO: misses = cold_misses + sum(hist[d] for d in [cache_size, max_dist)).
// Return misses / total_accesses.
double miss_rate_at_cache_size(const long* hist, int max_dist, long cold_misses,
                                long total_accesses, int cache_size) {
    (void)hist; (void)max_dist; (void)cold_misses; (void)total_accesses; (void)cache_size;
    // your code here
    return 0.0;
}
