#include <cstdio>
#include <cstdint>
#include "sol.hpp"

extern const int NUM_PARTICLES = 512;
extern const int NUM_FIELDS = 4;
const int CACHE_LINE_BYTES = 64;
const int CACHE_LINES = 32;  // direct-mapped, 32 * 64 = 2048-byte cache

namespace {
    int64_t cacheTag[CACHE_LINES];
    long long misses = 0;
    long long accesses = 0;

    void resetCache() {
        for (int i = 0; i < CACHE_LINES; i++) cacheTag[i] = -1;
        misses = 0;
        accesses = 0;
    }
}

// Deterministic direct-mapped cache model -- never real hardware counters,
// never wall-clock. `cacheTouch` is the ONLY way generateAoSoATrace can
// affect what gets measured.
void cacheTouch(int64_t byteAddr) {
    int64_t line = byteAddr / CACHE_LINE_BYTES;
    int set = (int)(line % CACHE_LINES);
    accesses++;
    if (cacheTag[set] != line) {
        misses++;
        cacheTag[set] = line;
    }
}

int main() {
    int tileWidths[] = {1, 2, 4, 8, 16, 32, 64};
    for (int tw : tileWidths) {
        resetCache();
        generateAoSoATrace(tw);
        printf("T%d accesses=%lld misses=%lld\n", tw, accesses, misses);
    }
    return 0;
}
