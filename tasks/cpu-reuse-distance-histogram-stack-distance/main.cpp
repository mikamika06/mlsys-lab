#include <cstdio>
#include "sol.hpp"

// FIXED driver. A fixed 12-access trace over 6 distinct lines (line
// numbers below, converted to byte addresses via LINE_BYTES), covering
// cold accesses, an immediate repeat (distance 0), and several longer
// reuse distances.
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 6;
constexpr int N = 12;
constexpr int TRACE_LINES[N] = {0, 1, 2, 1, 3, 0, 2, 4, 4, 5, 0, 1};

int main() {
    static long addrs[N];
    for (int i = 0; i < N; ++i) addrs[i] = static_cast<long>(TRACE_LINES[i]) * LINE_BYTES;

    static long hist[NUM_LINES + 1];
    stack_distance_histogram(addrs, N, LINE_BYTES, NUM_LINES, hist);

    printf("cold=%ld\n", hist[0]);
    for (int d = 0; d < NUM_LINES; ++d) {
        printf("dist%d=%ld\n", d, hist[1 + d]);
    }
    return 0;
}
