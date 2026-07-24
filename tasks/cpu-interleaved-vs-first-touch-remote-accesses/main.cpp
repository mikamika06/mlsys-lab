#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver: builds a deterministic trace over 4 NUMA nodes / 4
// threads (thread t pinned to node t). Two regions:
//   - PRIVATE: each thread touches its own exclusive 16-page (65536
//     byte) block, 4 times per page, and no other thread ever touches
//     it -- the case first-touch placement is built for.
//   - SHARED: a separate 16-page block that every thread touches once
//     per page, in thread order 0,1,2,3 (thread 0's full sweep happens
//     first) -- so under first-touch, thread 0 claims every shared
//     page, and threads 1-3's later sweeps are all remote.
int main() {
    const int NUM_NODES = 4;
    const int PRIVATE_PAGES = 16, PRIVATE_TOUCHES_PER_PAGE = 4;
    const int SHARED_PAGES = 16;

    std::vector<Access> trace;

    for (int t = 0; t < NUM_NODES; t++) {
        long base_page = (long)t * 64;  // 64-page spacing, no overlap
        for (int rep = 0; rep < PRIVATE_TOUCHES_PER_PAGE; rep++) {
            for (int p = 0; p < PRIVATE_PAGES; p++) {
                trace.push_back({t, (base_page + p) * PAGE_BYTES});
            }
        }
    }

    long shared_base_page = 256;  // well clear of every private block
    for (int t = 0; t < NUM_NODES; t++) {
        for (int p = 0; p < SHARED_PAGES; p++) {
            trace.push_back({t, (shared_base_page + p) * PAGE_BYTES});
        }
    }

    long first_touch_remote = 0, interleaved_remote = 0;
    count_remote_accesses(trace.data(), (int)trace.size(), NUM_NODES,
                           &first_touch_remote, &interleaved_remote);

    printf("n=%zu first_touch_remote=%ld interleaved_remote=%ld\n",
           trace.size(), first_touch_remote, interleaved_remote);
    return 0;
}
