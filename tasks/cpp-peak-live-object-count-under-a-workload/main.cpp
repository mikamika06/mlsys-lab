#include <cstdio>
#include "sol.hpp"

// FIXED driver: four alloc/free id sequences, matching four independent
// workloads. g_live/g_peak_live are reset before each. Prints the peak live
// count and peak_live * sizeof(Probe) (the real compiler's own struct size)
// for each.

static void run_case(const int* ids, const bool* is_alloc, int n) {
    g_live = 0;
    g_peak_live = 0;
    run_workload(ids, is_alloc, n);
    printf("%d %d\n", g_peak_live, g_peak_live * static_cast<int>(sizeof(Probe)));
}

int main() {
    {
        int ids[]      = {1, 2, 3, 1, 4, 2};
        bool allocs[]   = {true, true, true, false, true, false};
        run_case(ids, allocs, 6);
    }
    {
        int ids[]      = {10, 10, 20, 20};
        bool allocs[]   = {true, false, true, false};
        run_case(ids, allocs, 4);
    }
    {
        int ids[]      = {1, 2, 3, 4, 5};
        bool allocs[]   = {true, true, true, true, true};
        run_case(ids, allocs, 5);
    }
    {
        int ids[]      = {1, 2, 2, 1, 3};
        bool allocs[]   = {true, true, false, false, true};
        run_case(ids, allocs, 5);
    }
    return 0;
}
