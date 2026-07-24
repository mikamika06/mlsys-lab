#include <algorithm>
#include <vector>
#include "sol.hpp"

int classify_gather_strategy(const long* indices, int k, long elem_bytes) {
    // GATHER: read every request directly, in order.
    cache_reset();
    int gather_misses = 0;
    for (int i = 0; i < k; i++) {
        long addr = ORIG_BASE + indices[i] * elem_bytes;
        if (!touch(addr)) gather_misses++;
    }

    // DENSIFY: compact each distinct value once (ascending), then satisfy
    // every request from the compact scratch buffer by rank.
    cache_reset();
    std::vector<long> distinct(indices, indices + k);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

    int densify_misses = 0;
    for (long d : distinct) {
        long addr = ORIG_BASE + d * elem_bytes;
        if (!touch(addr)) densify_misses++;
    }
    for (int i = 0; i < k; i++) {
        long rank = std::lower_bound(distinct.begin(), distinct.end(), indices[i]) - distinct.begin();
        long addr = SCRATCH_BASE + rank * elem_bytes;
        if (!touch(addr)) densify_misses++;
    }

    return (gather_misses <= densify_misses) ? GATHER : DENSIFY;
}
