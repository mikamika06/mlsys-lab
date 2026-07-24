#include <cstdio>
#include "sol.hpp"

// FIXED driver: six (n_elements, reserve_first) scenarios, matching a
// no-reserve run against a reserve()'d run for a few sizes. Prints
// realloc_count, final_capacity, allocated_bytes (real sizeof(Item) times
// final_capacity) and pointers_valid for each.

static void run_case(int n, bool reserve_first) {
    GrowthResult r = grow_vector(n, reserve_first);
    long bytes = r.final_capacity * static_cast<long>(sizeof(Item));
    printf("%d %ld %ld %d\n", r.realloc_count, r.final_capacity, bytes, r.pointers_valid ? 1 : 0);
}

int main() {
    run_case(10, true);
    run_case(10, false);
    run_case(0, true);
    run_case(100, true);
    run_case(100, false);
    run_case(1024, true);
    return 0;
}
