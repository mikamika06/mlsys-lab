// Fixed driver: one fixed streaming-loop model (N, latency, per-iteration
// cost are all constants) run at several candidate prefetch distances.
// No timing, no randomness -- everything is fixed integer arithmetic.
#include "sol.hpp"
#include <cstdio>

int main() {
    const int N = 64;
    const int LATENCY_CYCLES = 180;
    const int CYCLES_PER_ITER = 20;
    // Sufficient distance is ceil(180 / 20) = 9. Below that: no benefit
    // at all (every access still stalls). At/above it: only the warm-up
    // iterations stall -- and an unnecessarily large distance pays for
    // more warm-up than it needs to.
    static const int DISTANCES[] = {1, 5, 8, 9, 12, 40};
    const int NUM_D = sizeof(DISTANCES) / sizeof(DISTANCES[0]);

    for (int i = 0; i < NUM_D; i++) {
        int d = DISTANCES[i];
        int stalls = count_stalls(N, d, LATENCY_CYCLES, CYCLES_PER_ITER);
        printf("distance=%d stalls=%d\n", d, stalls);
    }
    return 0;
}
