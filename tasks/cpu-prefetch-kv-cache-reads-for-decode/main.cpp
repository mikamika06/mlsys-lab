#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED deterministic cache model.
// 64-byte lines, 4 sets, 2-way -> 512-byte capacity (16 records of 32
// bytes each). T=32 records, so the working set is 2x the cache capacity.

namespace {
constexpr int LINE_BYTES = 64;
constexpr int SETS = 4;
constexpr int WAYS = 2;

struct Line { bool valid = false; long tag = -1; int lru = 0; };
Line g_cache[SETS][WAYS];
int g_clock = 0;
}  // namespace

void cache_reset() {
    for (int s = 0; s < SETS; s++)
        for (int w = 0; w < WAYS; w++) g_cache[s][w] = Line{};
    g_clock = 0;
}

bool touch(long byte_addr) {
    long line = byte_addr / LINE_BYTES;
    int set = (int)(((line % SETS) + SETS) % SETS);
    long tag = line / SETS;
    ++g_clock;

    for (int w = 0; w < WAYS; w++) {
        if (g_cache[set][w].valid && g_cache[set][w].tag == tag) {
            g_cache[set][w].lru = g_clock;
            return true;  // hit
        }
    }
    int victim = 0;
    for (int w = 1; w < WAYS; w++)
        if (g_cache[set][w].lru < g_cache[set][victim].lru) victim = w;
    g_cache[set][victim] = Line{true, tag, g_clock};
    return false;  // miss
}

int main() {
    const int T = 32, rec_bytes = 32;

    long naive_misses = simulate_decode_pass(T, rec_bytes, 0);
    int best_d = choose_best_prefetch_distance(T, rec_bytes, T - 1);
    long best_misses = simulate_decode_pass(T, rec_bytes, best_d);

    printf("naive_misses=%ld best_d=%d best_misses=%ld\n", naive_misses, best_d, best_misses);
    return 0;
}
