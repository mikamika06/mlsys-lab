#include "sol.hpp"
#include <unordered_set>

SimResult simulate_hash_map(const long* inserts, int n, double max_load_factor, int initial_buckets) {
    std::unordered_set<long> seen;
    long bucket_count = initial_buckets;
    int rehash_count = 0;

    for (int i = 0; i < n; i++) {
        long k = inserts[i];
        if (seen.find(k) == seen.end()) {
            long new_size = static_cast<long>(seen.size()) + 1;
            if (static_cast<double>(new_size) > max_load_factor * static_cast<double>(bucket_count)) {
                rehash_count++;
                bucket_count *= 2;
            }
            seen.insert(k);
        }
        // key already present: update-in-place, no size change, no rehash
    }

    SimResult r;
    r.rehash_count = rehash_count;
    r.hash_node_size = static_cast<int>(sizeof(HashNode));
    return r;
}
