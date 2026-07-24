#include "sol.hpp"

long lru_cache_misses(const long* addrs, int n, int num_sets, int ways) {
    // Small fixed-capacity model: each set's resident lines are stored
    // LRU-first..MRU-last in a small fixed row (num_sets, ways <= 16 in
    // this task).
    static const int MAX_SETS = 16;
    static const int MAX_WAYS = 16;
    static long lines[MAX_SETS][MAX_WAYS];
    int counts[MAX_SETS];
    for (int s = 0; s < num_sets; s++) counts[s] = 0;

    long misses = 0;
    for (int i = 0; i < n; i++) {
        long line = addrs[i] / LINE_BYTES;
        int s = static_cast<int>(line % num_sets);

        int found = -1;
        for (int w = 0; w < counts[s]; w++) {
            if (lines[s][w] == line) {
                found = w;
                break;
            }
        }

        if (found != -1) {
            // hit: shift everything after `found` down one, put line at MRU end.
            for (int w = found; w < counts[s] - 1; w++) lines[s][w] = lines[s][w + 1];
            lines[s][counts[s] - 1] = line;
        } else {
            misses++;
            if (counts[s] == ways) {
                // full: evict LRU (index 0), shift left, insert at MRU end.
                for (int w = 0; w < counts[s] - 1; w++) lines[s][w] = lines[s][w + 1];
                lines[s][counts[s] - 1] = line;
            } else {
                lines[s][counts[s]] = line;
                counts[s]++;
            }
        }
    }
    return misses;
}

void miss_triple(const long* addrs, int n, long* out3) {
    out3[0] = lru_cache_misses(addrs, n, 16, 1);
    out3[1] = lru_cache_misses(addrs, n, 4, 4);
    out3[2] = lru_cache_misses(addrs, n, 1, 16);
}
