// Fixed driver: three fixed byte-address traces (converted from fixed
// line-index sequences) run through simulate_lru at fixed capacities.
// No timing, no randomness.
#include "sol.hpp"
#include <cstdio>

namespace {
const int LINE_BYTES = 64;

// Trace A: pure streaming over 20 distinct lines, capacity 4 -- working
// set never fits, so there is no reuse at all.
const int TRACE_A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
const int LEN_A = sizeof(TRACE_A) / sizeof(TRACE_A[0]);
const int CAP_A = 4;

// Trace B: classic mixed-reuse LRU example, capacity 3.
const int TRACE_B[] = {0, 1, 2, 3, 0, 1, 4, 0, 1, 2, 3, 4};
const int LEN_B = sizeof(TRACE_B) / sizeof(TRACE_B[0]);
const int CAP_B = 3;

// Trace C: 3-line working set inside a capacity-4 cache -- after the
// first pass, every access hits.
const int TRACE_C[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
const int LEN_C = sizeof(TRACE_C) / sizeof(TRACE_C[0]);
const int CAP_C = 4;

void run(const char* name, const int* trace, int len, int capacity) {
    static long addrs[64];
    for (int i = 0; i < len; i++) addrs[i] = static_cast<long>(trace[i]) * LINE_BYTES;
    HitMiss r = simulate_lru(addrs, len, capacity, LINE_BYTES);
    printf("%s: hits=%ld misses=%ld\n", name, r.hits, r.misses);
}
} // namespace

int main() {
    run("traceA", TRACE_A, LEN_A, CAP_A);
    run("traceB", TRACE_B, LEN_B, CAP_B);
    run("traceC", TRACE_C, LEN_C, CAP_C);
    return 0;
}
