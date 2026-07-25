#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache (harness code, not learner code):
// 64-byte lines, 32 sets, 4-way -- 8192 bytes total capacity. Real hardware
// cache timing is not reproducible across machines, so this model -- not the
// CPU's actual cache -- is the sole source of every miss count printed below.
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

// Fixed test shape: 32 tokens, 8 heads, 16-float head dim, 2-byte elements.
// 2*D*E = 2*16*2 = 64, matching the "one head's K+V record is one cache
// line" invariant the lesson depends on.
static const int T = 32, H = 8, D = 16, E = 2;
static const long BASE = 0;

typedef long (*AddrFn)(long, int, int, int, int, int, int, int, int);

// Four alternative layouts, fixed here in the harness (not learner-owned),
// so the driver can show why THKD -- the learner's function -- wins.
static long tkhd_addr(long base, int T_, int H_, int D_, int E_, int t, int h, int k, int d) {
    long index = (((long)t * 2 + k) * H_ + h) * D_ + d;
    return base + index * E_;
}
static long tdhk_addr(long base, int T_, int H_, int D_, int E_, int t, int h, int k, int d) {
    long index = (((long)t * D_ + d) * H_ + h) * 2 + k;
    return base + index * E_;
}
static long htkd_addr(long base, int T_, int H_, int D_, int E_, int t, int h, int k, int d) {
    long index = (((long)h * T_ + t) * 2 + k) * D_ + d;
    return base + index * E_;
}
static long hktd_addr(long base, int T_, int H_, int D_, int E_, int t, int h, int k, int d) {
    long index = (((long)h * 2 + k) * T_ + t) * D_ + d;
    return base + index * E_;
}

static long run_write(AddrFn f) {
    reset_cache();
    int t = T - 1;  // append the newest token
    for (int h = 0; h < H; h++)
        for (int k = 0; k < 2; k++)
            for (int d = 0; d < D; d++)
                touch_byte(f(BASE, T, H, D, E, t, h, k, d));
    return miss_count();
}

static long run_read(AddrFn f) {
    reset_cache();
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int k = 0; k < 2; k++)
                for (int d = 0; d < D; d++)
                    touch_byte(f(BASE, T, H, D, E, t, h, k, d));
    return miss_count();
}

int main() {
    const char* names[5] = {"THKD", "TKHD", "TDHK", "HTKD", "HKTD"};
    AddrFn fns[5] = {thkd_addr, tkhd_addr, tdhk_addr, htkd_addr, hktd_addr};

    long total[5];
    for (int i = 0; i < 5; i++) {
        long w = run_write(fns[i]);
        long r = run_read(fns[i]);
        total[i] = w + r;
        printf("%s write_misses=%ld read_misses=%ld total=%ld\n", names[i], w, r, total[i]);
    }

    int best = 0;
    for (int i = 1; i < 5; i++) if (total[i] < total[best]) best = i;
    printf("best=%s\n", names[best]);
    return 0;
}
