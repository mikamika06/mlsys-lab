#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Builds a real reuse-distance histogram (via a plain O(n^2)
// scan, obviously correct for this small trace) from a fixed access trace:
// a working set of 8 distinct block ids, round-robin cycled 20 times. Then
// queries the learner's tail-sum miss-rate function at several cache
// sizes to trace out the miss-rate curve's knee at the working-set size.

namespace {
constexpr int MAX_DIST = 32;

void build_histogram(const std::vector<int>& trace, long* hist, int max_dist,
                      long* cold_misses, long* total_accesses) {
    int n = (int)trace.size();
    long cold = 0;
    for (int d = 0; d < max_dist; d++) hist[d] = 0;

    for (int i = 0; i < n; i++) {
        int b = trace[i];
        int j = -1;
        for (int k = i - 1; k >= 0; k--) {
            if (trace[k] == b) { j = k; break; }
        }
        if (j < 0) {
            cold++;
            continue;
        }
        // Count distinct addresses in the open interval (j, i).
        std::vector<int> seen;
        int distinct = 0;
        for (int k = j + 1; k < i; k++) {
            bool found = false;
            for (int v : seen) if (v == trace[k]) { found = true; break; }
            if (!found) { seen.push_back(trace[k]); distinct++; }
        }
        if (distinct < max_dist) hist[distinct]++;
    }
    *cold_misses = cold;
    *total_accesses = n;
}
}  // namespace

int main() {
    // Working set of 8 distinct blocks, cycled 20 times -> 160 accesses.
    std::vector<int> trace;
    for (int cycle = 0; cycle < 20; cycle++)
        for (int b = 0; b < 8; b++) trace.push_back(b);

    long hist[MAX_DIST];
    long cold_misses, total_accesses;
    build_histogram(trace, hist, MAX_DIST, &cold_misses, &total_accesses);

    int queries[5] = {1, 4, 7, 8, 16};
    for (int c : queries) {
        double mr = miss_rate_at_cache_size(hist, MAX_DIST, cold_misses, total_accesses, c);
        printf("C=%d miss_rate=%.9f\n", c, mr);
    }
    return 0;
}
