#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model is what the driver
// grades against, not the CPU's real cache.
namespace {

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

Level CACHE(64, 32, 2);  // 64B lines, 32 sets, 2-way -> 4096 bytes

}  // namespace

void cache_reset() { CACHE = Level(64, 32, 2); }
void touch(long byte_addr) { CACHE.access(byte_addr); }
long cache_misses() { return CACHE.misses; }

// FIXED driver: n=24 matrices of 8-byte doubles (three 24x24 matrices =
// 13824 bytes -- 3.4x the cache's 4096-byte capacity).
int main() {
    const int n = 24;
    char out[3][4] = {{0}};

    rank_matmul_orders(n, out);

    printf("ranking=%s,%s,%s\n", out[0], out[1], out[2]);
    return 0;
}
