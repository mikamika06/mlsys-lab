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

// HARNESS baseline (not learner code): a plain triple-nested ijk matmul,
// no tiling, purely for comparison against the learner's tiled version.
static void naive_matmul(int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                touch(a_addr(N, i, k));
                touch(b_addr(N, k, j));
                touch(c_addr(N, i, j));
            }
        }
    }
}

// FIXED driver. Multiplies two 48x48 matrices (three 48x48 float32
// matrices together are 27648 bytes -- 3.4x the cache's capacity) once
// the naive way and once the tiled way, on two independent fresh caches,
// and prints both miss counts.
int main() {
    const int N = 48, T = 8;

    naive_matmul(N);
    long naive_misses = CACHE.misses;

    CACHE = Level(64, 32, 4);  // fresh cache for the second pass
    tiled_matmul(N, T);
    long tiled_misses = CACHE.misses;

    printf("naive_misses=%ld tiled_misses=%ld\n", naive_misses, tiled_misses);
    return 0;
}
