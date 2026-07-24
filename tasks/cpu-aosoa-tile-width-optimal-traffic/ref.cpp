#include "sol.hpp"

void generateAoSoATrace(int tileWidth) {
    const int fieldBytes = 4;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int tileIdx = i / tileWidth;
        int withinTile = i % tileWidth;
        long long tileBytes = (long long)tileWidth * NUM_FIELDS * fieldBytes;
        for (int f = 0; f < 3; f++) {  // x, y, z -- not mass
            long long fieldOffset = (long long)f * tileWidth * fieldBytes;
            long long addr = tileIdx * tileBytes + fieldOffset + (long long)withinTile * fieldBytes;
            cacheTouch(addr);
        }
    }
}
