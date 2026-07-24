#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model is what the driver
// grades against, not the CPU's real cache. Line size and associativity
// are pinned; total CAPACITY is reconfigured between scenarios below by
// changing the number of sets.
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

static Level CACHE(64, 2, 4);  // reconfigured before every scenario below

void touch(long byte_addr) { CACHE.access(byte_addr); }

// FIXED driver. Cache-oblivious-transposes a 64x64 matrix (in+out together
// = 32768 bytes) against 4 DIFFERENT cache capacities -- 512B, 2048B,
// 8192B, 32768B -- 64-byte lines and 4-way associativity fixed throughout,
// only the number of sets changes. co_transpose() itself never sees any
// of these numbers: the same implementation runs, unmodified, at every
// size. Prints the miss count and miss rate (misses / total accesses) for
// each of the 4 sizes, forming a 4-point miss-rate curve (MRC).
int main() {
    const int N = 64;                              // 64x64 float matrix
    const long total_accesses = 2L * N * N;         // 1 read + 1 write / element
    const int sizes[4] = {512, 2048, 8192, 32768};  // bytes; 64B lines, 4-way

    for (int k = 0; k < 4; k++) {
        int nsets = sizes[k] / (64 * 4);
        CACHE = Level(64, nsets, 4);
        co_transpose(N);
        double rate = (double)CACHE.misses / (double)total_accesses;
        printf("size=%d misses=%ld rate=%.4f\n", sizes[k], CACHE.misses, rate);
    }
    return 0;
}
