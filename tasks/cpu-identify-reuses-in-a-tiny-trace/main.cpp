#include <cstdint>
#include <cstdio>

#include "sol.hpp"

// FIXED driver: a small deterministic address trace over a working set of
// 10 distinct byte addresses (8 bytes apart), generated with a seeded LCG
// so the trace is reproducible across every run and every machine.
static uint32_t g_rng = 7919u;
static uint32_t next_rand() {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) & 0xFFFFFFu;
}

int main() {
    const int N = 40;
    const int WORKING_SET = 10;  // 10 distinct addresses -> plenty of reuse
    static long trace[N];

    for (int i = 0; i < N; i++) {
        trace[i] = (long)(next_rand() % WORKING_SET) * 8;
    }

    long long reuses = count_reuses(trace, N);
    printf("%lld\n", reuses);
    return 0;
}
