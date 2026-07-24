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

static Level CACHE(64, 32, 4);  // 64B lines, 32 sets, 4-way -> 8192 bytes

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): a plain row-by-row nested-loop
// transpose, no blocking, purely for comparison against the learner's
// blocked version.
static void naive_transpose(int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            touch(in_addr(N, i, j));
            touch(out_addr(N, j, i));
        }
    }
}

// FIXED driver. Transposes a 64x64 matrix (32768 bytes for `in` + `out`
// together -- 4x the cache's capacity) once the naive way and once the
// blocked way, on two independent fresh caches, and prints both miss
// counts.
int main() {
    const int N = 64, B = 8;

    naive_transpose(N);
    long naive_misses = CACHE.misses;

    CACHE = Level(64, 32, 4);  // fresh cache for the second pass
    blocked_transpose(N, B);
    long blocked_misses = CACHE.misses;

    printf("naive_misses=%ld blocked_misses=%ld\n", naive_misses, blocked_misses);
    return 0;
}
