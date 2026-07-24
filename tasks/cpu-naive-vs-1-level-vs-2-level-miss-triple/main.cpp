#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache: 64-byte lines, 16 sets, 4-way
// -> 4096 bytes total. Real hardware cache behaviour is not reproducible
// across machines, so this model is what the driver grades against, not
// the CPU's real cache.
struct Level {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long misses = 0;

    Level(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    void access(long address) {
        long line = address / line_bytes;
        auto& s = sets[(int)(line % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        misses++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

static Level CACHE(64, 16, 4);

void touch(long byte_addr) { CACHE.access(byte_addr); }
void reset_cache() { CACHE = Level(64, 16, 4); }
long miss_count() { return CACHE.misses; }

// FIXED driver, two fixed matmul scenarios. Each of A, B, C is N x N
// float; for N=64 that's 16384 bytes per matrix (49152 combined), well
// over the 4096-byte cache, so tiling should visibly help. Matrices are
// placed back-to-back: A at 0, B right after A, C right after B.
int main() {
    struct Scenario { int N, tile1, tile2; };
    static const Scenario scenarios[] = {
        {64, 16, 8},
        {32, 8, 4},
    };

    for (const auto& s : scenarios) {
        long a_base = 0;
        long b_base = a_base + (long)s.N * s.N * 4;
        long c_base = b_base + (long)s.N * s.N * 4;
        long out[3] = {0, 0, 0};
        matmul_miss_triple(s.N, s.tile1, s.tile2, a_base, b_base, c_base, out);
        printf("N=%d tile1=%d tile2=%d naive=%ld tiled1=%ld tiled2=%ld\n",
               s.N, s.tile1, s.tile2, out[0], out[1], out[2]);
    }
    return 0;
}
