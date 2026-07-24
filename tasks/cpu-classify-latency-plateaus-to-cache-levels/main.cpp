#include <cstdint>
#include <cstdio>
#include "sol.hpp"

// FIXED driver. Deterministic, no rand()/time()/hardware counters: a
// hand-rolled 64-bit LCG with a fixed seed perturbs 4 true per-level
// latencies by +-15% and hands each noisy sample to the learner's
// classify_plateau(). Prints "<index> <latency> <level>" per sample, one
// line each. The latency values themselves come only from this driver (they
// do not depend on solve.cpp), so the exact_match gate isolates the
// classification labels.

namespace {

// True per-level latencies: arbitrary but well-separated simulated cycles,
// each level roughly 3x its inner neighbour, matching the shape of the real
// L1/L2/L3/DRAM ladder.
constexpr double kTrueLatency[4] = {4.0, 12.0, 36.0, 140.0};
constexpr int kSamplesPerLevel = 8;

// Deterministic LCG (splitmix64-style), fixed seed — no rand(), no time().
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    double next01() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) / static_cast<double>(1ULL << 53);
    }
};

}  // namespace

int main() {
    Lcg rng(20260724ULL);
    int idx = 0;
    for (int level = 0; level < 4; ++level) {
        for (int s = 0; s < kSamplesPerLevel; ++s) {
            double noise = (rng.next01() * 2.0 - 1.0) * 0.15;  // +-15%
            double latency = kTrueLatency[level] * (1.0 + noise);
            CacheLevel got = classify_plateau(latency);
            printf("%d %.6f %d\n", idx, latency, static_cast<int>(got));
            ++idx;
        }
    }
    return 0;
}
