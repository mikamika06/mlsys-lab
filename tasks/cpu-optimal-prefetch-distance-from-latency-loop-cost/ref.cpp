#include "sol.hpp"

int prefetch_distance(int mem_latency, int loop_body_cycles) {
    return (mem_latency + loop_body_cycles - 1) / loop_body_cycles;
}

int count_stalls(int n, int mem_latency, int loop_body_cycles, int distance) {
    int stalls = 0;
    for (int i = 0; i < n; ++i) {
        int issue_iter = i - distance;
        if (issue_iter < 0) issue_iter = 0;
        long long landing_cycle = (long long)issue_iter * loop_body_cycles + mem_latency;
        long long consume_cycle = (long long)i * loop_body_cycles;
        if (landing_cycle > consume_cycle) ++stalls;
    }
    return stalls;
}
