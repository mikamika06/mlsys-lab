#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache: 64-byte lines, 8 sets, 4-way
// -> 2048 bytes (512 floats' worth of lines resident at once), much
// smaller than the 4096-float table below. Real hardware cache behaviour
// is not reproducible across machines, so this model is what the driver
// grades against, not the CPU's real cache.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

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

static Level CACHE(64, 8, 4);

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): gathers in the exact ORIGINAL
// (unsorted) index order, purely for comparison against the learner's
// version.
static void naive_gather(const float* table, const int* indices, int n, float* output) {
    for (int i = 0; i < n; i++) {
        output[i] = table[indices[i]];
        touch(table_addr(indices[i]));
    }
}

static float TABLE[4096];
static int INDICES[2000];
static float OUT_NAIVE[2000];
static float OUT_SORTED[2000];

static long checksum(const float* out, int n) {
    long s = 0;
    for (int i = 0; i < n; i++) s += (long)out[i] * (long)(i + 1);
    return s;
}

// FIXED driver. A 4096-float table (16384 bytes, 8x the cache) and 2000
// DISTINCT, scattered indices (indices[i] = (i*37) mod 4096 -- 37 is
// coprime with 4096, so this is a genuine permutation of [0,4096), not a
// repeating pattern). Runs the harness's naive (original-order) gather
// and the learner's gather_sorted, each against its own fresh cache, and
// prints both miss counts plus a position-weighted checksum of the
// learner's output (sensitive to values ending up at the wrong index).
int main() {
    const int table_len = 4096;
    const int n = 2000;
    for (int k = 0; k < table_len; k++) TABLE[k] = (float)k;
    for (int i = 0; i < n; i++) INDICES[i] = (i * 37) % table_len;

    naive_gather(TABLE, INDICES, n, OUT_NAIVE);
    long naive_misses = CACHE.misses;

    CACHE = Level(64, 8, 4);
    gather_sorted(TABLE, table_len, INDICES, n, OUT_SORTED);
    long sorted_misses = CACHE.misses;

    printf("naive_misses=%ld sorted_misses=%ld checksum=%ld\n",
           naive_misses, sorted_misses, checksum(OUT_SORTED, n));
    return 0;
}
