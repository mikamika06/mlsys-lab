// Fixed driver: two fixed 6-thread x 3-node x capacity-2 scenarios. No
// timing, no randomness -- every access-count matrix is a hardcoded
// table.
#include "sol.hpp"
#include <cstdio>

namespace {
const int T = 6, N = 3, CAPACITY = 2;

// Case 1: threads 0, 1, and 2 all most want node 0, but only 2 of the 3
// fit -- the right move is to displace thread 1 (it loses almost
// nothing, 90 -> 85, by settling for node 1 instead), not thread 0 or
// thread 2 (each would lose 75-90 by being displaced).
const long CASE1[T * N] = {
    100, 10, 5,   // thread 0
    90,  85, 5,   // thread 1
    80,  5,  5,   // thread 2
    5,   5,  90,  // thread 3
    5,   90, 5,   // thread 4
    5,   5,  5,   // thread 5
};

// Case 2: no contention at all -- exactly 2 threads clearly prefer each
// node, so a plain per-thread greedy pick already lands on the optimum.
const long CASE2[T * N] = {
    100, 5,  5,   // thread 0
    90,  5,  5,   // thread 1
    5,   100, 5,  // thread 2
    5,   90,  5,  // thread 3
    5,   5,  100, // thread 4
    5,   5,  90,  // thread 5
};
} // namespace

int main() {
    long r1 = min_remote_accesses(T, N, CAPACITY, CASE1);
    long r2 = min_remote_accesses(T, N, CAPACITY, CASE2);
    printf("case1_min_remote=%ld\n", r1);
    printf("case2_min_remote=%ld\n", r2);
    return 0;
}
