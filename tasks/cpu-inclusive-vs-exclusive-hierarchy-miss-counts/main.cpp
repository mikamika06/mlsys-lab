#include <cstdio>
#include "sol.hpp"

// FIXED driver. Deterministic pseudo-random trace (hand-rolled LCG, no
// rand()/time) over NUM_LINES distinct cache lines -- more lines than
// L2_WAYS, so both levels feel real eviction pressure.
constexpr int NUM_LINES = 24;
constexpr int N = 400;

int main() {
    static long addrs[N];
    unsigned state = 12345u;
    for (int i = 0; i < N; ++i) {
        state = state * 1103515245u + 12345u;
        int idx = static_cast<int>((state >> 16) % NUM_LINES);
        addrs[i] = static_cast<long>(idx) * LINE_BYTES;
    }

    long out2[2] = {0, 0};
    hierarchy_miss_counts(addrs, N, out2);

    printf("inclusive_misses=%ld\n", out2[0]);
    printf("exclusive_misses=%ld\n", out2[1]);
    return 0;
}
