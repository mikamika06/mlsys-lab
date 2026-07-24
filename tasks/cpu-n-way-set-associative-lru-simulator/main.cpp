#include <cstdint>
#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Every trace and every cache geometry is hand-built (no
// rand()/time()), so the expected miss counts are easy to hand-verify (see
// task.md).

namespace {

void run(const char* name, const std::vector<uint64_t>& lines,
         int line_bytes, int sets, int ways) {
    std::vector<uint64_t> addrs;
    addrs.reserve(lines.size());
    for (uint64_t l : lines) addrs.push_back(l * static_cast<uint64_t>(line_bytes));

    int misses = count_misses(addrs.data(), static_cast<int>(addrs.size()), line_bytes, sets, ways);
    printf("%s sets=%d ways=%d n=%zu misses=%d\n", name, sets, ways, addrs.size(), misses);
}

}  // namespace

int main() {
    // P: direct-mapped (ways=1), 4 sets. Lines 0,1,2,3 each map to their
    // own set; swept once then repeated once more. No two lines ever
    // share a set, so the repeat pass is all hits.
    run("direct_mapped", {0, 1, 2, 3, 0, 1, 2, 3}, 64, 4, 1);

    // Q: 3 addresses that all hash to the SAME set (stride = sets), 2-way,
    // round-robin 4 times. 3 distinct addresses contending for 2 ways ->
    // every access after the very first lap still misses (see
    // cpu-classify-misses-cold-capacity-conflict for the full derivation).
    run("conflict_thrash", {100, 104, 108, 100, 104, 108, 100, 104, 108, 100, 104, 108}, 64, 4, 2);

    // R: fully-associative (sets=1), 8 ways. Lines 0..9 swept once (10
    // distinct lines, only 8 ways -> the two oldest get evicted), then 8
    // and 9 (the two most recently touched) are repeated -- both still
    // resident, so both hit.
    run("fully_assoc_reuse", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9}, 64, 1, 8);

    return 0;
}
