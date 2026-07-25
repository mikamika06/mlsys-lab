#include "sol.hpp"
#include <cstdio>

// FIXED driver, two scenarios of 5 kernels each. Scenario 1:
// peak_flops=200 GFLOP/s, peak_bandwidth=50 GB/s (ridge point = 4
// FLOP/byte). Scenario 2: peak_flops=400 GFLOP/s, peak_bandwidth=80 GB/s
// (ridge point = 5 FLOP/byte). All FLOP/byte figures given in giga-units
// so the ratios stay simple; no two kernels in either scenario tie on
// attainable throughput.
int main() {
    {
        double flops[5] = {1.0, 2.0, 0.5, 1.2, 0.3};
        double bytes[5] = {1.0, 0.1, 0.25, 0.4, 0.6};
        int rank[5];
        rank_kernels_by_attainable_perf(flops, bytes, 5, 200.0, 50.0, rank);
        printf("scenario=1 rank=%d %d %d %d %d\n",
               rank[0], rank[1], rank[2], rank[3], rank[4]);
    }
    {
        double flops[5] = {0.8, 3.0, 0.4, 1.0, 0.2};
        double bytes[5] = {0.8, 0.3, 0.1, 0.5, 1.0};
        int rank[5];
        rank_kernels_by_attainable_perf(flops, bytes, 5, 400.0, 80.0, rank);
        printf("scenario=2 rank=%d %d %d %d %d\n",
               rank[0], rank[1], rank[2], rank[3], rank[4]);
    }
    return 0;
}
