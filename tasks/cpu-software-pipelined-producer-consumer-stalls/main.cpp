#include <cstdio>
#include "sol.hpp"

// FIXED driver. Every (n, latency) pair is hand-picked (no rand()/time()).

namespace {
void run(const char* name, long long n, long long latency) {
    StallCounts s = modeled_stall_counts(n, latency);
    printf("%s n=%lld latency=%lld naive_stalls=%lld pipelined_stalls=%lld\n",
           name, n, latency, s.naive_stalls, s.pipelined_stalls);
}
}  // namespace

int main() {
    run("single_iteration", 1, 4);
    run("ten_iterations", 10, 4);
    run("thousand_iterations", 1000, 6);
    run("zero_latency", 5, 0);
    run("zero_iterations", 0, 8);
    return 0;
}
