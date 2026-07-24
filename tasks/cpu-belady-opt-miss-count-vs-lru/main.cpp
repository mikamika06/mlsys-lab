#include <cstdint>
#include <cstdio>

#include "sol.hpp"

// FIXED driver: a deterministic reference string over a working set of 6
// page ids, run through a fully-associative cache of capacity 4 -- large
// enough that greedy/short-sighted eviction choices (e.g. "evict slot 0",
// or LRU on a working set bigger than the cache) provably miss more often
// than the clairvoyant OPT policy does.
static uint32_t g_rng = 4242u;
static uint32_t next_rand() {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) & 0xFFFFFFu;
}

int main() {
    const int N = 50;
    const int WORKING_SET = 6;
    const int CAPACITY = 4;
    static int refs[N];

    // Mostly cycle through a small working set (creates real reuse), with
    // occasional jumps to keep it from being perfectly periodic.
    int cursor = 0;
    for (int i = 0; i < N; i++) {
        if (next_rand() % 5 == 0) {
            cursor = (int)(next_rand() % WORKING_SET);
        } else {
            cursor = (cursor + 1) % WORKING_SET;
        }
        refs[i] = cursor;
    }

    int misses = belady_opt_misses(refs, N, CAPACITY);
    printf("%d\n", misses);
    return 0;
}
