#include <cstdint>
#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Deterministic, no rand()/time(): a splitmix64-style LCG with
// a fixed seed generates the access trace. 4 threads split across 2 NUMA
// nodes; 10 pages, but the trace only ever touches pages 0..8, so page 9
// must come back owned by node -1 (never touched).

namespace {

struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int range(int lo, int hi) {
        uint64_t span = static_cast<uint64_t>(hi - lo) + 1;
        return lo + static_cast<int>(next() % span);
    }
};

}  // namespace

int main() {
    constexpr int kNumThreads = 4;
    constexpr int kNumPages = 10;
    constexpr int kTouchedPages = 9;  // pages 0..8 are touched; page 9 never is
    constexpr int kN = 40;

    const int node_of_thread[kNumThreads] = {0, 0, 1, 1};

    Lcg rng(20260724ULL);
    std::vector<int> thread_of_access(kN), page_of_access(kN);
    for (int i = 0; i < kN; ++i) {
        thread_of_access[i] = rng.range(0, kNumThreads - 1);
        page_of_access[i] = rng.range(0, kTouchedPages - 1);
    }

    std::vector<int> owner(kNumPages, -99);
    first_touch_owner(thread_of_access.data(), page_of_access.data(), kN,
                       node_of_thread, kNumThreads, kNumPages, owner.data());

    for (int i = 0; i < kN; ++i) {
        printf("access %d thread=%d page=%d\n", i, thread_of_access[i], page_of_access[i]);
    }
    for (int p = 0; p < kNumPages; ++p) {
        printf("page %d owner=%d\n", p, owner[p]);
    }
    return 0;
}
