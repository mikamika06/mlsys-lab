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
long miss_count() { return CACHE.misses; }

// FIXED driver. Deterministic xorshift-filled X (32x512) and W (32x512)
// float matrices (65536 bytes each -- 16x the 8192-byte cache), a fresh
// cache, one call to einsum_bij, and the resulting checksum + miss
// count.
int main() {
    const int B = 32, I = 32, J = 512;
    static float X[B * J], W[I * J], Y[B * I];

    unsigned long s = 12345;
    auto rnd = [&]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((s % 1000) / 1000.0f - 0.5f);
    };
    for (int k = 0; k < B * J; k++) X[k] = rnd();
    for (int k = 0; k < I * J; k++) W[k] = rnd();

    long x_base = 0;
    long w_base = x_base + (long)B * J * 4;
    long y_base = w_base + (long)I * J * 4;

    reset_cache();
    float checksum = einsum_bij(B, I, J, x_base, w_base, y_base, X, W, Y);
    long misses = miss_count();

    printf("checksum=%.6f misses=%ld\n", checksum, misses);
    return 0;
}
