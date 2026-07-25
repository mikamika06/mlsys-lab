#include <cstdio>
#include <vector>
#include <list>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model is what the driver
// grades against, not the CPU's real cache.
struct Cache {
    int line_bytes, nsets, ways;
    std::vector<std::list<long>> sets;
    long miss_count = 0;

    Cache(int lb, int ns, int w) : line_bytes(lb), nsets(ns), ways(w), sets(ns) {}

    void access(long addr) {
        long line = addr / line_bytes;
        auto& s = sets[(int)(((line % nsets) + nsets) % nsets)];
        for (auto it = s.begin(); it != s.end(); ++it) {
            if (*it == line) { s.erase(it); s.push_front(line); return; }
        }
        miss_count++;
        if ((int)s.size() >= ways) s.pop_back();
        s.push_front(line);
    }
};

static Cache CACHE(64, 32, 4);  // 64B lines, 32 sets, 4-way -> 8192 bytes

void reset_cache() { CACHE = Cache(64, 32, 4); }
void touch(long byte_addr) { CACHE.access(byte_addr); }
long misses() { return CACHE.miss_count; }

constexpr int N = 4096;
constexpr int VEC_WIDTH = 16;   // 16 floats * 4 bytes == 64 bytes == 1 line
constexpr int ELEM_BYTES = 4;

// 12-bit bit-reversal permutation (N == 2^12): a classic worst-locality
// access order used to stress caches (same idea as FFT bit-reversal
// reordering). idx_scrambled visits the SAME 4096 elements as the
// contiguous pass, in an order that scatters accesses to any given
// 64-byte line (16 consecutive indices) far apart in time.
static int bit_reverse12(int x) {
    int r = 0;
    for (int b = 0; b < 12; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

int main() {
    static int idx_scrambled[N];
    for (int k = 0; k < N; ++k) idx_scrambled[k] = bit_reverse12(k);

    reset_cache();
    long contig_uops = contiguous_load(0, N, VEC_WIDTH, ELEM_BYTES);
    long contig_misses = misses();

    reset_cache();
    long gather_uops = gather_load(0, idx_scrambled, N, ELEM_BYTES);
    long gather_misses = misses();

    printf("contig_uops=%ld contig_misses=%ld\n", contig_uops, contig_misses);
    printf("gather_uops=%ld gather_misses=%ld\n", gather_uops, gather_misses);
    return 0;
}
