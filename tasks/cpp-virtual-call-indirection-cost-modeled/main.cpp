#include <cstdio>
#include "sol.hpp"

int main() {
    // A fixed, deterministic call trace with runs of the same receiver and
    // repeated methods, so caching can eliminate some indirect loads.
    const int N = 20;
    const int obj[N]  = {0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 1, 0, 4, 4, 2, 5, 5, 5, 6, 6};
    const int slot[N] = {0, 0, 1, 0, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0};

    long naive  = naive_virtual_loads(obj, slot, N);
    long cached = cached_virtual_loads(obj, slot, N);
    long devirt = devirtualized_loads(obj, slot, N);

    printf("naive=%ld\n", naive);
    printf("cached=%ld\n", cached);
    printf("devirt=%ld\n", devirt);
    printf("saved=%ld\n", naive - cached);
    return 0;
}
