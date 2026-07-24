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

// HARNESS baseline (not learner code): a plain column-major traversal of
// a row-major matrix, no interchange -- purely for comparison against
// the learner's row-major version.
static void column_major_traverse(int N) {
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < N; row++) {
            touch(elem_addr(N, row, col));
        }
    }
}

// FIXED driver. Sweeps a 64x64 matrix of doubles (32768 bytes -- 4x the
// cache's capacity) once column-major and once row-major, on two
// independent fresh caches, and prints both miss counts.
int main() {
    const int N = 64;

    column_major_traverse(N);
    long column_major_misses = CACHE.misses;

    CACHE = Level(64, 32, 4);  // fresh cache for the second pass
    row_major_traverse(N);
    long row_major_misses = CACHE.misses;

    printf("column_major_misses=%ld row_major_misses=%ld\n", column_major_misses, row_major_misses);
    return 0;
}
