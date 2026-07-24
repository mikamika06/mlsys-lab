#include <cmath>
#include "sol.hpp"

CacheLevel classify_plateau(double latency_cycles) {
    constexpr double kTrue[4] = {4.0, 12.0, 36.0, 140.0};
    const double mid12 = std::sqrt(kTrue[0] * kTrue[1]);
    const double mid23 = std::sqrt(kTrue[1] * kTrue[2]);
    const double mid34 = std::sqrt(kTrue[2] * kTrue[3]);

    if (latency_cycles < mid12) return CacheLevel::L1;
    if (latency_cycles < mid23) return CacheLevel::L2;
    if (latency_cycles < mid34) return CacheLevel::L3;
    return CacheLevel::DRAM;
}
