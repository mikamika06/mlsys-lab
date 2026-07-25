#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache
// behaviour isn't reproducible across machines, so this model -- not
// the CPU's actual cache -- is the sole source of every miss count the
// driver prints.
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

static Level CACHE(64, 16, 4);  // 64B lines, 16 sets, 4-way -> 4096 bytes

void touch(long byte_addr) { CACHE.access(byte_addr); }

// HARNESS baseline (not learner code): the naive gather, processing
// requests in ORIGINAL i-order with no reordering at all.
static long naive_gather_misses(const int* idx, int n) {
    CACHE = Level(64, 16, 4);
    for (int i = 0; i < n; i++) touch((long)idx[i] * (long)sizeof(float));
    return CACHE.misses;
}

// FIXED driver. 4096 floats of data (16KB, 4x the cache's capacity).
// 8192 gather requests: the first 4096 sweep every element once in
// order, the second 4096 request the SAME 4096 elements again in the
// same order -- so by the time a naive in-order pass revisits an
// element, hundreds of unrelated lines have been touched since, well
// past the cache's capacity, and every "repeat" misses again too.
int main() {
    const int DSIZE = 4096, N = 8192;

    std::vector<float> data(DSIZE);
    for (int i = 0; i < DSIZE; i++) data[i] = (float)i * 0.5f + 1.0f;

    std::vector<int> idx(N);
    for (int i = 0; i < DSIZE; i++) idx[i] = i;
    for (int i = 0; i < DSIZE; i++) idx[DSIZE + i] = i;

    long naive_misses = naive_gather_misses(idx.data(), N);

    std::vector<float> out(N, 0.0f);
    CACHE = Level(64, 16, 4);
    segmented_gather(data.data(), DSIZE, idx.data(), N, out.data());
    long learner_misses = CACHE.misses;

    double checksum = 0.0;
    for (int i = 0; i < N; i++) checksum += out[i];

    printf("naive_misses=%ld learner_misses=%ld checksum=%.6f\n", naive_misses, learner_misses, checksum);
    return 0;
}
