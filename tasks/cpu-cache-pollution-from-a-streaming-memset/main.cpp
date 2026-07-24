#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model is what the driver
// grades against, not the CPU's real cache.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    // Returns true iff this access is a MISS (line was not resident).
    bool access(long addr) {
        long line = addr / line_bytes;
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return false; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
        return true;
    }
};

static Level CACHE(64, 16, 4);  // 64B lines, 16 sets, 4-way -> 4096 bytes

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): a streaming (non-temporal) memset.
// Real NT stores write through a write-combining buffer straight to
// memory and never allocate a line in the CPU cache, so the correct
// model of one is to touch nothing at all.
static void streaming_memset(long /*base*/, long /*nbytes*/, int /*line_bytes*/) {
    // intentionally empty: non-temporal stores never touch the cache model.
}

// Re-touches the K resident "useful" lines and returns how many of them
// MISS -- i.e. how many were evicted by whatever ran in between.
static int count_evicted(long useful_base, int line_bytes, int K) {
    int evicted = 0;
    for (int k = 0; k < K; k++) {
        if (CACHE.access(line_addr(useful_base, line_bytes, k))) evicted++;
    }
    return evicted;
}

// FIXED driver. A small "useful" 8-line working set (512 bytes) is warmed
// into a fresh 4096-byte cache, then a big 8192-byte bulk write (2x the
// cache's capacity) runs over a disjoint region -- once as a streaming
// (non-temporal) memset, once as your temporal_memset -- after which the
// driver re-touches the useful lines and counts how many were evicted.
int main() {
    const int LINE = 64, K = 8;          // 8-line "useful" working set
    const long USEFUL_BASE = 0;          // lines 0..7   -> sets 0..7
    const long MEM_BASE = 4096;          // line 64 first -> also set 0
    const long MEM_BYTES = 128L * LINE;  // 128 lines, 8x the cache's capacity

    // --- streaming (non-temporal) pass, fresh cache ---
    CACHE = Level(64, 16, 4);
    for (int k = 0; k < K; k++) touch(line_addr(USEFUL_BASE, LINE, k));  // warm useful set
    streaming_memset(MEM_BASE, MEM_BYTES, LINE);
    int evicted_streaming = count_evicted(USEFUL_BASE, LINE, K);

    // --- temporal pass, fresh cache ---
    CACHE = Level(64, 16, 4);
    for (int k = 0; k < K; k++) touch(line_addr(USEFUL_BASE, LINE, k));  // warm useful set
    temporal_memset(MEM_BASE, MEM_BYTES, LINE);
    int evicted_temporal = count_evicted(USEFUL_BASE, LINE, K);

    printf("evicted_streaming=%d evicted_temporal=%d\n", evicted_streaming, evicted_temporal);
    return 0;
}
