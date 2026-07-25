#include <cstdio>
#include <list>
#include <vector>
#include "sol.hpp"

// Deterministic set-associative LRU cache. Real hardware cache behaviour
// is not reproducible across machines, so this model -- not the CPU's
// actual cache -- is what the driver grades against.
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

// HARNESS traversal (not learner code): for every row, touch row `r`,
// column 0 of every one of the K matrices (a "reduce across the stack"
// pass), then immediately do the SAME K touches again (a second pass over
// the identical addresses, e.g. a follow-up statistic computed from the
// first) -- so a conflict-aliased layout evicts row data before the
// second pass can reuse it, while a non-aliased layout keeps it resident.
long run_stack_traversal(int K, int M, int ld, int elem_bytes, int line_bytes, int nsets, int ways) {
    Level cache(line_bytes, nsets, ways);
    for (int r = 0; r < M; r++) {
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k < K; k++) {
                long addr = (long)k * M * ld * elem_bytes + (long)r * ld * elem_bytes;
                cache.access(addr);
            }
        }
    }
    return cache.misses;
}
}  // namespace

// FIXED driver. 8 matrices, 8 rows x 16 cols of 4-byte elements, cache:
// 64-byte lines, 8 sets, 4-way (2048 bytes). Unpadded (ld=16), the
// per-matrix stride (8*16*4=512 bytes) is an exact multiple of the
// cache's 512-byte set-period, so all 8 matrices' row-r alias one set.
int main() {
    const int K = 8, M = 8, N = 16, elem_bytes = 4, line_bytes = 64, nsets = 8, ways = 4;

    long naive_misses = run_stack_traversal(K, M, N, elem_bytes, line_bytes, nsets, ways);

    int ld = choose_padded_ld(N, M, elem_bytes, line_bytes, nsets);
    long padded_misses = run_stack_traversal(K, M, ld, elem_bytes, line_bytes, nsets, ways);

    printf("ld=%d naive_misses=%ld padded_misses=%ld\n", ld, naive_misses, padded_misses);
    return 0;
}
