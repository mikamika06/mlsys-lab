#include <algorithm>
#include <array>
#include <cstdio>

#include "sol.hpp"

// FIXED driver: four archetypal memory access patterns, reduced to the
// {n_misses, window, latency, chained} parameters the model needs.
// Pattern indices: 0=pointer_chase, 1=sequential, 2=strided,
// 3=scatter_gather. Derivation (64-element traversal, 64-byte lines,
// 4-byte elements -> 16 elements/line, latency = 100 cycles/miss):
//
//   0 pointer_chase:  each of the 64 hops is its own miss (a linked-list
//                      style traversal rarely reuses a line), and the
//                      NEXT address is only known once the current node's
//                      data comes back -> chained = true.
//   1 sequential:     unit-stride over 64 elements touches only
//                      ceil(64/16) = 4 distinct lines; independent
//                      (address is `base + i`, never data-dependent);
//                      window = 8 (a typical out-of-order MSHR count).
//   2 strided:        stride == one full line, so all 64 accesses are
//                      distinct-line misses; independent; same window=8
//                      MSHR budget as sequential (still one scalar loop,
//                      still bounded by the reorder buffer).
//   3 scatter_gather: also 64 independent misses, but expressed as
//                      several concurrently-issued streams (e.g. a
//                      software-pipelined or SIMD gather), which can
//                      keep twice as many MSHRs busy at once -> window=16.
struct PatternSpec { int n_misses, window, latency; bool chained; };

static const PatternSpec PATTERNS[4] = {
    {64, 8, 100, true},    // 0 pointer_chase
    {4, 8, 100, false},    // 1 sequential
    {64, 8, 100, false},   // 2 strided
    {64, 16, 100, false},  // 3 scatter_gather
};

int main() {
    std::array<MlpResult, 4> results;
    for (int i = 0; i < 4; i++) {
        const PatternSpec& p = PATTERNS[i];
        results[i] = simulate_mlp(p.n_misses, p.window, p.latency, p.chained);
        printf("cycles[%d]=%lld mlp_x1000[%d]=%lld\n", i, results[i].cycles, i, results[i].mlp_x1000);
    }

    std::array<int, 4> order = {0, 1, 2, 3};
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return results[a].mlp_x1000 < results[b].mlp_x1000;
    });

    printf("rank:");
    for (int idx : order) printf(" %d", idx);
    printf("\n");
    return 0;
}
