#include "sol.hpp"

StallCounts modeled_stall_counts(long long n, long long latency) {
    StallCounts s;
    s.naive_stalls = n * latency;
    s.pipelined_stalls = (n > 0) ? latency : 0;
    return s;
}
