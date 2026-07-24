#include <cstdint>
#include <cstdio>

#include "sol.hpp"

// FIXED driver: build a deterministic sequence of 30 allocation requests
// with cycling power-of-two alignments and pseudo-random sizes chosen so
// most of them do NOT already land on an aligned boundary (forcing real
// padding most of the time), then ask for the total wasted bytes.
static uint32_t g_rng = 777u;
static uint32_t next_rand() {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) & 0xFFFFFFu;
}

int main() {
    const int N = 30;
    static int sizes[N];
    static int alignments[N];

    const int align_cycle[5] = {4, 8, 16, 32, 64};
    for (int i = 0; i < N; i++) {
        alignments[i] = align_cycle[i % 5];
        sizes[i] = 1 + (int)(next_rand() % 97);  // 1..97, deliberately odd-ish
    }

    long wasted = total_wasted_bytes(sizes, alignments, N);
    printf("%ld\n", wasted);
    return 0;
}
