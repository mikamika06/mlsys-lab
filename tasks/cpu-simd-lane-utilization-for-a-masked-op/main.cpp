#include <cstdio>
#include "sol.hpp"

// FIXED driver. width=8, n=32 (4 groups): group0 fully active, group1
// has 2 of 8 active, group2 fully inactive (skipped), group3 has 5 of 8
// active.
int main() {
    const int width = 8;
    const int n = 32;
    bool mask[n] = {
        true,  true,  true,  true,  true,  true,  true,  true,   // group 0: 8/8
        true,  true,  false, false, false, false, false, false,  // group 1: 2/8
        false, false, false, false, false, false, false, false,  // group 2: 0/8 (skipped)
        true,  true,  true,  true,  true,  false, false, false,  // group 3: 5/8
    };

    double util = simd_lane_utilization(mask, n, width);
    printf("utilization=%.6f\n", util);
    return 0;
}
