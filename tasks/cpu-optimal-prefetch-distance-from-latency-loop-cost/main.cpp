#include <cstdio>
#include "sol.hpp"

// FIXED driver. Derives the optimal prefetch distance for two
// (latency, loop-cost) pairs, then feeds the first pair's distance --
// and, for contrast, a deliberately too-small fixed distance of 1 --
// into the stall simulator over a long-running loop, to show that the
// derived distance bounds stalls to a fixed startup cost while an
// undersized distance stalls on every single iteration.
int main() {
    int d1 = prefetch_distance(120, 30);   // ceil(120/30) == 4
    int d2 = prefetch_distance(121, 30);   // ceil(121/30) == 5
    int d3 = prefetch_distance(1, 1);      // ceil(1/1)    == 1

    const int N = 1000;
    int stalls_optimal = count_stalls(N, 120, 30, d1);
    int stalls_naive = count_stalls(N, 120, 30, 1);

    printf("d1=%d d2=%d d3=%d\n", d1, d2, d3);
    printf("stalls_optimal=%d stalls_naive=%d\n", stalls_optimal, stalls_naive);
    return 0;
}
