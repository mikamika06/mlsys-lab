// Fixed driver: builds one deterministic byte-address trace and feeds it
// through miss_triple. No timing, no randomness -- the trace is a fixed
// table.
#include "sol.hpp"
#include <cstdio>

const int LINE_BYTES = 64;

int main() {
    // 8 distinct cache lines, each 16 lines apart -- i.e. all congruent
    // mod 16, a classic power-of-two-stride aliasing pattern: they all
    // hash to the SAME direct-mapped set and the SAME 4-way set. The
    // access pattern itself has real locality: lines 0 and 1 are "hot"
    // (reused constantly, like loop-carried accumulators), lines 2..7
    // are "cold" (touched once per pass, like a large strided array).
    const int NUM_LINES = 8;
    long base_line[NUM_LINES];
    for (int k = 0; k < NUM_LINES; k++) base_line[k] = 3 + 16 * k;

    static const int block[] = {0, 1, 0, 1, 2, 0, 1, 3, 0, 1, 4,
                                 0, 1, 5, 0, 1, 6, 0, 1, 7};
    const int BLOCK_LEN = sizeof(block) / sizeof(block[0]);
    const int PASSES = 3;
    const int N = BLOCK_LEN * PASSES;

    static long addrs[N];
    int idx = 0;
    for (int p = 0; p < PASSES; p++) {
        for (int b = 0; b < BLOCK_LEN; b++) {
            addrs[idx++] = base_line[block[b]] * LINE_BYTES;
        }
    }

    long triple[3];
    miss_triple(addrs, N, triple);
    printf("direct_mapped=%ld four_way=%ld fully_assoc=%ld\n", triple[0], triple[1], triple[2]);
    return 0;
}
