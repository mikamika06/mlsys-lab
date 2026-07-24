#include "sol.hpp"

// BUG: ignores tileWidth entirely and treats the data as one big flat SoA
// array (as if tileWidth == NUM_PARTICLES always) -- field f's whole column
// is one contiguous run of NUM_PARTICLES values, with no per-tile
// structure. This only happens to match the real AoSoA layout when
// tileWidth == NUM_PARTICLES; for every smaller tile width it computes the
// wrong address.
void generateAoSoATrace(int tileWidth) {
    const int fieldBytes = 4;
    (void)tileWidth;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        for (int f = 0; f < 3; f++) {  // x, y, z
            long long addr = (long long)f * NUM_PARTICLES * fieldBytes + (long long)i * fieldBytes;
            cacheTouch(addr);
        }
    }
}
