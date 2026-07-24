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
void touch(long addr) { CACHE.access(addr); }
void touch_nt(long addr) { (void)addr; /* bypasses the cache entirely */ }
long miss_count() { return CACHE.misses; }

namespace {
constexpr long H_BASE = 0, H_BYTES = 2048;      // 32 lines, 1 per set
constexpr long BUF_BASE = 65536;                 // clear of H, set-aligned
}  // namespace

long run_workload(long working_set_bytes, bool use_nt, bool reused_soon) {
    reset_cache();
    for (long off = 0; off < H_BYTES; off += 4) touch(H_BASE + off);

    for (long off = 0; off < working_set_bytes; off += 4) {
        if (use_nt) touch_nt(BUF_BASE + off);
        else touch(BUF_BASE + off);
    }

    long cost = 0;
    long before = miss_count();
    for (long off = 0; off < H_BYTES; off += 4) touch(H_BASE + off);
    cost += miss_count() - before;

    if (reused_soon) {
        long before2 = miss_count();
        for (long off = 0; off < working_set_bytes; off += 4) touch(BUF_BASE + off);
        cost += miss_count() - before2;
    }
    return cost;
}

// FIXED driver: 5 buffer sizes (below, at, and above the 8192-byte
// cache) crossed with reused_soon in {false, true} -- 10 scenarios --
// each asking the candidate's nt_stores_help() for a verdict.
int main() {
    const long sizes[] = {2048, 4096, 8192, 16384, 32768};
    for (long sz : sizes) {
        for (int r = 0; r < 2; r++) {
            bool reused = r != 0;
            bool helps = nt_stores_help(sz, reused);
            printf("size=%ld reused=%d nt_helps=%d\n", sz, reused ? 1 : 0, helps ? 1 : 0);
        }
    }
    return 0;
}
