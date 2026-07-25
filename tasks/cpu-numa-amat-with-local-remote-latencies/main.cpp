#include <cstdio>
#include "sol.hpp"

// FIXED driver: seven deterministic NUMA configurations, including the
// zero-miss-rate, all-local and all-remote edge cases.
int main() {
    struct TC {
        double l3_hit, miss_rate, local_dram, remote_base;
        int hops;
        double per_hop, local_frac;
    };
    static const TC cases[] = {
        {10.0, 0.05, 80.0, 100.0, 2, 25.0, 0.7},
        {8.0, 0.10, 70.0, 120.0, 3, 20.0, 0.5},
        {12.0, 0.02, 90.0, 110.0, 1, 30.0, 0.9},
        {6.0, 0.15, 60.0, 150.0, 4, 15.0, 0.3},
        {10.0, 0.00, 80.0, 100.0, 2, 25.0, 0.7},   // zero miss rate edge
        {10.0, 0.05, 80.0, 100.0, 1, 25.0, 1.0},   // all-local edge
        {10.0, 0.05, 80.0, 100.0, 1, 25.0, 0.0},   // all-remote edge
    };

    for (const auto& tc : cases) {
        double amat = compute_numa_amat(tc.l3_hit, tc.miss_rate, tc.local_dram,
                                         tc.remote_base, tc.hops, tc.per_hop, tc.local_frac);
        printf("%.9f\n", amat);
    }
    return 0;
}
