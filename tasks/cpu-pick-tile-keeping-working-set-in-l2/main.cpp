#include <cstdio>
#include "sol.hpp"

// Deterministic direct-mapped cache model (harness code, not learner
// code): 64-byte lines, 64 sets -> 4096 bytes total, 1 way per set.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 64;  // 4096 bytes total, direct-mapped
long g_tag[NUM_LINES];
bool g_valid[NUM_LINES];
long g_misses;
}  // namespace

void reset_cache() {
    for (int i = 0; i < NUM_LINES; i++) g_valid[i] = false;
    g_misses = 0;
}

void touch_byte(long addr) {
    long line = addr / LINE_BYTES;
    int set = (int)(line % NUM_LINES);
    if (g_valid[set] && g_tag[set] == line) return;  // hit
    g_misses++;
    g_valid[set] = true;
    g_tag[set] = line;
}

long miss_count() { return g_misses; }

// FIXED driver. tile_b0=16 -> 3*16*16*4 = 3072 bytes (fits the 4096-byte
// L2). tile_b1=32 -> 3*32*32*4 = 12288 bytes (does not fit). 5 passes.
int main() {
    const int tile_b0 = 16;
    const int tile_b1 = 32;
    const int passes = 5;

    long out_misses[2] = {-1, -1};  // sentinel: an empty starter leaves this untouched
    int winner = pick_resident_tile(tile_b0, tile_b1, passes, out_misses);

    printf("winner=%d misses0=%ld misses1=%ld\n", winner, out_misses[0], out_misses[1]);
    return 0;
}
