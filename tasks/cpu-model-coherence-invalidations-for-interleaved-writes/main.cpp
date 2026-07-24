#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver: 4 cores, each incrementing its own private counter 100
// times, interleaved round-robin (core 0, 1, 2, 3, 0, 1, 2, 3, ...) --
// two layouts of the exact same logical work:
//   - UNPADDED: all 4 counters packed into 16 bytes, so all 4 addresses
//     fall in the SAME 64-byte cache line.
//   - PADDED: each counter placed 64 bytes apart, one per cache line.
int main() {
    const int NUM_CORES = 4;
    const int ITERS = 100;

    std::vector<WriteEvent> unpadded;
    for (int it = 0; it < ITERS; it++)
        for (int c = 0; c < NUM_CORES; c++)
            unpadded.push_back({c, (long)c * 4});

    std::vector<WriteEvent> padded;
    for (int it = 0; it < ITERS; it++)
        for (int c = 0; c < NUM_CORES; c++)
            padded.push_back({c, (long)c * 64});

    long u = count_invalidations(unpadded.data(), (int)unpadded.size());
    long p = count_invalidations(padded.data(), (int)padded.size());

    printf("n=%d unpadded_invalidations=%ld padded_invalidations=%ld\n",
           ITERS * NUM_CORES, u, p);
    return 0;
}
