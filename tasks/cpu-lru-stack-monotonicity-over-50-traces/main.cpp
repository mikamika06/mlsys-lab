#include <cstdio>
#include <vector>
#include "sol.hpp"

int main() {
    const int NUM_TRACES = 50;
    const int TRACE_LEN = 200;
    const int NBLOCKS = 64;
    const int CAPS[6] = {1, 2, 4, 8, 16, 32};
    const int NUM_CAPS = 6;

    // Deterministic fixed-seed generator (no rand(), no time/clock) --
    // same 50 traces every run.
    unsigned long state = 0x2545F4914F6CDD1DULL;
    auto next_id = [&]() -> int {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (int)((state >> 33) % (unsigned long)NBLOCKS);
    };

    int monotonic_count = 0;
    long total_miss_sum = 0;

    std::vector<int> trace(TRACE_LEN);
    for (int t = 0; t < NUM_TRACES; t++) {
        for (int i = 0; i < TRACE_LEN; i++) trace[i] = next_id();

        long misses[6];
        for (int k = 0; k < NUM_CAPS; k++) {
            misses[k] = lru_miss_count(trace.data(), TRACE_LEN, CAPS[k]);
            total_miss_sum += misses[k];
        }

        bool monotonic = true;
        for (int k = 1; k < NUM_CAPS; k++) {
            if (misses[k] > misses[k - 1]) { monotonic = false; break; }
        }
        if (monotonic) monotonic_count++;
    }

    printf("monotonic_count=%d total_miss_sum=%ld\n", monotonic_count, total_miss_sum);
    return 0;
}
