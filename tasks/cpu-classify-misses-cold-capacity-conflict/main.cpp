#include <cstdint>
#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. All 4 traces are built from simple deterministic formulas
// (no rand()/time()) with disjoint address ranges per building block, so the
// three trace "shapes" below stay easy to reason about. Cache params are
// pinned: 64-byte lines, 4 sets, 2 ways -> 8 lines / 512 bytes total.

namespace {

constexpr int kLineBytes = 64;
constexpr int kSets = 4;
constexpr int kWays = 2;

std::vector<uint64_t> trace_cold(int base_line, int count) {
    std::vector<uint64_t> t;
    t.reserve(count);
    for (int i = 0; i < count; ++i)
        t.push_back(static_cast<uint64_t>(base_line + i) * kLineBytes);
    return t;
}

// `distinct` addresses, all mapping to the SAME set (stride is a multiple of
// kSets so (base + k*stride) % kSets never changes), repeated `repeats`
// times back to back.
std::vector<uint64_t> trace_conflict(int base_line, int stride, int distinct, int repeats) {
    std::vector<uint64_t> t;
    t.reserve(distinct * repeats);
    for (int r = 0; r < repeats; ++r)
        for (int k = 0; k < distinct; ++k)
            t.push_back(static_cast<uint64_t>(base_line + k * stride) * kLineBytes);
    return t;
}

// `distinct` consecutive lines (spread evenly across all sets), swept
// `passes` times.
std::vector<uint64_t> trace_capacity(int base_line, int distinct, int passes) {
    std::vector<uint64_t> t;
    t.reserve(distinct * passes);
    for (int p = 0; p < passes; ++p)
        for (int i = 0; i < distinct; ++i)
            t.push_back(static_cast<uint64_t>(base_line + i) * kLineBytes);
    return t;
}

void run_trace(const char* name, const std::vector<uint64_t>& addrs) {
    MissCounts m = classify_misses(addrs.data(), static_cast<int>(addrs.size()),
                                    kLineBytes, kSets, kWays);
    printf("%s cold=%d capacity=%d conflict=%d\n", name, m.cold, m.capacity, m.conflict);
}

}  // namespace

int main() {
    // Trace A: pure cold -- 20 unique lines, single pass, no repeats.
    run_trace("A", trace_cold(0, 20));

    // Trace B: pure conflict -- 3 addresses that all map to the same set,
    // repeated 4 times (12 accesses). They easily fit an 8-line
    // fully-associative cache but thrash a 2-way set.
    run_trace("B", trace_conflict(100, kSets, 3, 4));

    // Trace C: pure capacity -- 16 unique lines spread evenly across all 4
    // sets, swept twice (32 accesses). The working set (16 lines) does not
    // fit even a fully-associative 8-line cache.
    run_trace("C", trace_capacity(1000, 16, 2));

    // Trace D: mixed -- cold-only lines, a conflicting block, and a
    // capacity-thrashing block, back to back in one trace (disjoint address
    // ranges so the building blocks stay legible).
    std::vector<uint64_t> mixed;
    auto append = [&mixed](const std::vector<uint64_t>& v) {
        mixed.insert(mixed.end(), v.begin(), v.end());
    };
    append(trace_cold(2000, 5));
    append(trace_conflict(3000, kSets, 3, 4));
    append(trace_capacity(4000, 16, 2));
    run_trace("D", mixed);

    return 0;
}
