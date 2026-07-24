#include "sol.hpp"
#include <cstdio>

struct Config { int line_words; int base_latency; int cycles_per_word; };

// FIXED driver. For each of 4 memory-system configurations, sweeps every
// possible requested ("critical") word in the line and prints the
// time-to-first-useful-word BOTH without critical-word-first (burst
// always starts at word 0) AND with it (burst starts at the requested
// word), plus the cycles saved.
int main() {
    static const Config configs[] = {
        {8, 40, 4},    // 8-word line, 40-cycle DRAM latency, 4 cyc/word burst
        {16, 40, 2},   // wider line, faster per-word transfer
        {8, 100, 1},   // high-latency memory, fast burst
        {4, 20, 8},    // short line, slow burst
    };

    for (const auto& cfg : configs) {
        for (int target = 0; target < cfg.line_words; target++) {
            long without_cwf = time_to_word(cfg.line_words, 0, target,
                                             cfg.base_latency, cfg.cycles_per_word);
            long with_cwf = time_to_word(cfg.line_words, target, target,
                                          cfg.base_latency, cfg.cycles_per_word);
            printf("W=%d target=%d without_cwf=%ld with_cwf=%ld saved=%ld\n",
                   cfg.line_words, target, without_cwf, with_cwf,
                   without_cwf - with_cwf);
        }
    }
    return 0;
}
