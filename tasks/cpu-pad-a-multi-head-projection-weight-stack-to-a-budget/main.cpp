#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache, reconfigurable between
// reset_cache() calls. Real hardware cache behaviour is not reproducible
// across machines, so this model is what the driver grades against, not
// the CPU's real cache.
struct Level {
    int line_bytes = 64, nsets = 1, ways = 1;
    std::vector<std::list<long>> sets{1};
    long misses = 0;

    void access(long addr) {
        long line = addr / line_bytes;
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

static Level CACHE;

void touch(long byte_addr) { CACHE.access(byte_addr); }
long miss_count() { return CACHE.misses; }
void reset_cache(int line_bytes, int sets, int ways) {
    CACHE = Level();
    CACHE.line_bytes = line_bytes;
    CACHE.nsets = sets;
    CACHE.ways = ways;
    CACHE.sets.assign(sets, {});
    CACHE.misses = 0;
}

// HARNESS-owned (not learner code): the REAL kernel this padding decision
// is protecting -- gather all H heads' value at row r, then gather them
// again right after (e.g. compute-then-normalize), for every row of an
// R-row stack, on ONE continuous cache session (no reset between rows).
static long run_full_stack(int H, int row_bytes, int pad, int R,
                            int line_bytes, int sets, int ways) {
    reset_cache(line_bytes, sets, ways);
    long stride = (long)row_bytes + pad;
    for (int r = 0; r < R; r++) {
        for (int h = 0; h < H; h++) touch((long)h * stride + (long)r * 4);
        for (int h = 0; h < H; h++) touch((long)h * stride + (long)r * 4);
    }
    return miss_count();
}

// FIXED driver, two scenarios. H=16 heads, row_bytes=256 (a 64-row stack
// of floats per head -- power-of-two dims, like a real per-head
// projection stack), against a 2048-byte (64B line, 8 sets, 4-way) cache
// in scenario 1, and a smaller 8-head / 512-byte-row / 1024-byte cache in
// scenario 2.
int main() {
    struct Scenario { int H, row_bytes, line_bytes, sets, ways, max_pad, R; };
    static const Scenario scenarios[] = {
        {16, 256, 64, 8, 4, 252, 64},
        {8, 512, 64, 4, 4, 508, 32},
    };

    for (const auto& s : scenarios) {
        int pad = choose_padding_bytes(s.H, s.row_bytes, s.line_bytes, s.sets,
                                        s.ways, s.max_pad);
        long total_misses = run_full_stack(s.H, s.row_bytes, pad, s.R,
                                            s.line_bytes, s.sets, s.ways);
        printf("H=%d row_bytes=%d line=%d sets=%d ways=%d pad=%d total_misses=%ld\n",
               s.H, s.row_bytes, s.line_bytes, s.sets, s.ways, pad, total_misses);
    }
    return 0;
}
