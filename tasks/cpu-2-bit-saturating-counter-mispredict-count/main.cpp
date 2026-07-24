#include <cstdint>
#include <cstdio>

#include "sol.hpp"

// FIXED driver: build a deterministic, round-robin-interleaved trace of
// branch events across 4 branches with distinct, realistic behaviours --
// mostly-taken, mostly-not-taken, strictly alternating (the classic case a
// 2-bit counter handles poorly), and pseudo-random -- then ask for the
// mispredict count and print it.
static uint32_t g_rng = 12345u;
static uint32_t next_rand() {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) & 0xFFFFFFu;
}

int main() {
    const int NUM_BRANCHES = 4;
    const int LEN_PER_BRANCH = 100;
    const int N = NUM_BRANCHES * LEN_PER_BRANCH;

    static int branch_ids[N];
    static int outcomes[N];

    int per_branch_i[NUM_BRANCHES] = {0, 0, 0, 0};
    for (int i = 0; i < N; i++) {
        int b = i % NUM_BRANCHES;
        branch_ids[i] = b;
        int k = per_branch_i[b]++;
        int outcome;
        if (b == 0) {
            outcome = (next_rand() % 100) < 90 ? 1 : 0;   // mostly taken
        } else if (b == 1) {
            outcome = (next_rand() % 100) < 10 ? 1 : 0;   // mostly not-taken
        } else if (b == 2) {
            outcome = k % 2;                               // strictly alternating
        } else {
            outcome = next_rand() % 2;                     // ~50/50
        }
        outcomes[i] = outcome;
    }

    int mispredicts = count_mispredicts(branch_ids, outcomes, N, NUM_BRANCHES);
    printf("%d\n", mispredicts);
    return 0;
}
