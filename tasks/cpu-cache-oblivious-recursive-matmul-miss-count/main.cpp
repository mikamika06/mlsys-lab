#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache (harness code, not learner
// code): 64-byte lines, 32 sets, 4-way -- 8192 bytes total capacity.
// Real hardware cache timing is not reproducible across machines, so
// this model -- not the CPU's actual cache -- is the sole source of
// every miss count the driver prints.
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

void reset_cache() { CACHE = Level(64, 32, 4); }
void touch_byte(long addr) { CACHE.access(addr); }
long miss_count() { return CACHE.misses; }

// FIXED driver. Three 64x64 float matrices (16384 bytes each -- 49152
// bytes together, 6x the 8192-byte cache) laid back to back in
// simulated address space. Multiplies them once with naive_matmul and
// once with recursive_matmul, each against its own fresh cache, and
// prints both miss counts.
int main() {
    const int N = 64;
    long a_base = 0;
    long b_base = a_base + (long)N * N * 4;
    long c_base = b_base + (long)N * N * 4;

    reset_cache();
    naive_matmul(N, a_base, b_base, c_base);
    long naive_misses = miss_count();

    reset_cache();
    recursive_matmul(N, a_base, b_base, c_base);
    long recursive_misses = miss_count();

    printf("naive_misses=%ld recursive_misses=%ld\n", naive_misses, recursive_misses);
    return 0;
}
