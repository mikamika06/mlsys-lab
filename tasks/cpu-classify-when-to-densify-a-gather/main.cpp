#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED deterministic cache model.
//
// Cache params (pinned): 64-byte lines, 8 sets, 4-way associative, LRU
// eviction -> capacity = 64*8*4 = 2048 bytes.

namespace {
constexpr int LINE_BYTES = 64;
constexpr int SETS = 8;
constexpr int WAYS = 4;

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

namespace {
// Repeat a fixed pattern `reps` times into one index vector.
std::vector<long> repeated(const std::vector<long>& pattern, int reps) {
    std::vector<long> out;
    out.reserve(pattern.size() * reps);
    for (int r = 0; r < reps; r++)
        for (long v : pattern) out.push_back(v);
    return out;
}
}  // namespace

int main() {
    constexpr long ELEM_BYTES = 4;  // float

    // A: dense cluster [0,16), cycled 8x (128 accesses, 16 distinct) --
    //    already cache-friendly (spans exactly one 64-byte line).
    std::vector<long> pattern_a;
    for (int i = 0; i < 16; i++) pattern_a.push_back(i);
    std::vector<long> a = repeated(pattern_a, 8);

    // B: 8 indices spaced exactly SETS*LINE_BYTES/ELEM_BYTES apart (all
    //    alias the SAME cache set), cycled 10x -- classic thrash pattern
    //    with heavy reuse.
    std::vector<long> pattern_b = {0, 128, 256, 384, 512, 640, 768, 896};
    std::vector<long> b = repeated(pattern_b, 10);

    // C: sequential stream [0,128), each index touched exactly once -- no
    //    reuse at all.
    std::vector<long> c;
    for (int i = 0; i < 128; i++) c.push_back(i);

    // D: same 8 set-aliasing indices as B, but touched once each -- no
    //    reuse.
    std::vector<long> d = pattern_b;

    // E: 6 set-aliasing indices (same stride as B), cycled 3x -- moderate
    //    reuse.
    std::vector<long> pattern_e = {0, 128, 256, 384, 512, 640};
    std::vector<long> e = repeated(pattern_e, 3);

    int la = classify_gather_strategy(a.data(), (int)a.size(), ELEM_BYTES);
    int lb = classify_gather_strategy(b.data(), (int)b.size(), ELEM_BYTES);
    int lc = classify_gather_strategy(c.data(), (int)c.size(), ELEM_BYTES);
    int ld = classify_gather_strategy(d.data(), (int)d.size(), ELEM_BYTES);
    int le = classify_gather_strategy(e.data(), (int)e.size(), ELEM_BYTES);

    printf("%d %d %d %d %d\n", la, lb, lc, ld, le);
    return 0;
}
