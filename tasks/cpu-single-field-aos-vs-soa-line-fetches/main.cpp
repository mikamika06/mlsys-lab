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

static Level CACHE(64, 8, 4);  // 64B lines, 8 sets, 4-way -> 2048 bytes

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): reads one field out of a
// 64-byte-per-record AoS array, no rearrangement -- purely for
// comparison against the learner's SoA version.
static void aos_field_touch(int N, long aos_base, int record_bytes, int field_offset) {
    for (int i = 0; i < N; i++) {
        touch(aos_base + (long)i * record_bytes + field_offset);
    }
}

// FIXED driver. 2000 records, 64 bytes each (one full cache line per
// record -- a realistic wide record, e.g. an embedding row), reading a
// field near the middle of each. Runs the AoS baseline and the learner's
// SoA version, each against its own fresh cache, and prints both miss
// counts.
int main() {
    const int N = 2000;
    const int record_bytes = 64;
    const int field_offset = 20;
    const long aos_base = 0;
    const long soa_base = (long)N * record_bytes;  // right after the AoS region

    aos_field_touch(N, aos_base, record_bytes, field_offset);
    long aos_misses = CACHE.misses;

    CACHE = Level(64, 8, 4);  // fresh cache for the second pass
    soa_field_touch(N, soa_base);
    long soa_misses = CACHE.misses;

    printf("aos_misses=%ld soa_misses=%ld\n", aos_misses, soa_misses);
    return 0;
}
