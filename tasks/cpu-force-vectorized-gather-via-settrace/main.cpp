#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache: 64-byte lines, 8 sets, 4-way
// -> 2048 bytes resident at once, far smaller than the 16384-byte table
// below. Real hardware cache behaviour is not reproducible across
// machines, so this model -- not any real CPU cache -- is what the
// driver grades against. Also counts every touch() call, regardless of
// hit/miss.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;
    long touches = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    void access(long addr) {
        touches++;
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

// HARNESS baseline (not learner code): a naive scalar gather that
// touches memory once per OUTPUT POSITION, even when the index value
// repeats -- for comparison against the learner's deduping version.
static void naive_gather(const float* table, const int* indices, int n, float* output) {
    for (int i = 0; i < n; i++) {
        output[i] = table[indices[i]];
        touch(table_addr(indices[i]));
    }
}

static float TABLE[4096];
static int INDICES[4000];
static float OUT_NAIVE[4000];
static float OUT_DEDUP[4000];

static long checksum(const float* out, int n) {
    long s = 0;
    for (int i = 0; i < n; i++) s += (long)out[i] * (long)(i + 1);
    return s;
}

// FIXED driver. A 4096-float table (16384 bytes, 8x the modeled cache).
// Only 64 DISTINCT index values ever appear: HOT[j] = j*61 mod 4096 (61
// is coprime with 4096, so all 64 are genuinely distinct and spread
// across the whole table); indices[i] = HOT[i % 64] cycles through them
// for 4000 accesses -- a "hot working set reused across a long batch"
// gather pattern, like repeated vocab ids in a token stream. Runs the
// harness's naive (always-touch) gather and the learner's gather_dedup,
// each against its own fresh cache, and prints both touch/miss counts
// plus a position-weighted checksum of the learner's output (sensitive
// to a value landing at the wrong index).
int main() {
    const int table_len = 4096;
    const int n = 4000;
    const int nhot = 64;
    int HOT[nhot];
    for (int j = 0; j < nhot; j++) HOT[j] = (j * 61) % table_len;
    for (int k = 0; k < table_len; k++) TABLE[k] = (float)k;
    for (int i = 0; i < n; i++) INDICES[i] = HOT[i % nhot];

    naive_gather(TABLE, INDICES, n, OUT_NAIVE);
    long naive_touches = CACHE.touches;
    long naive_misses = CACHE.misses;

    CACHE = Level(64, 8, 4);
    gather_dedup(TABLE, table_len, INDICES, n, OUT_DEDUP);
    long dedup_touches = CACHE.touches;
    long dedup_misses = CACHE.misses;

    printf("naive_touches=%ld naive_misses=%ld dedup_touches=%ld dedup_misses=%ld checksum=%ld\n",
           naive_touches, naive_misses, dedup_touches, dedup_misses, checksum(OUT_DEDUP, n));
    return 0;
}
