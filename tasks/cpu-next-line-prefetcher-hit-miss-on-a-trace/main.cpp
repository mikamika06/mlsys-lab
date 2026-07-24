#include <cstdio>
#include "sol.hpp"

// FIXED driver: a sequential run (0..3), a jump to a distant region
// (50..52), then a jump back into the original stream (4, 5). Cache holds
// only 3 lines, so the jumps evict lines the prefetcher had readied.
int main() {
    int trace[9] = {0, 1, 2, 3, 50, 51, 52, 4, 5};
    const int n = 9;
    const int cache_lines = 3;

    int hits = -1, misses = -1;  // sentinel: an empty starter leaves these untouched
    simulate_next_line_prefetch(trace, n, cache_lines, &hits, &misses);

    printf("hits=%d misses=%d\n", hits, misses);
    return 0;
}
