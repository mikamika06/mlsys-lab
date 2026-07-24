#include "sol.hpp"
#include <bit>
#include <cstdint>

static inline int clamp_keep(int keep_bits) {
    if (keep_bits < 0) return 0;
    if (keep_bits > 23) return 23;
    return keep_bits;
}

void quantize_mantissa(float* x, int n, int keep_bits) {
    const uint32_t drop = uint32_t(23 - clamp_keep(keep_bits));   // low bits to zero
    const uint32_t mask = ~((uint32_t(1) << drop) - 1u);          // clear low `drop` bits
    for (int i = 0; i < n; i++) {
        // strict-aliasing-safe: reinterpret via std::bit_cast, not *(uint32_t*)&x[i]
        uint32_t u = std::bit_cast<uint32_t>(x[i]);
        u &= mask;
        x[i] = std::bit_cast<float>(u);
    }
}

int count_bits_lost(const float* x, int n, int keep_bits) {
    const uint32_t drop = uint32_t(23 - clamp_keep(keep_bits));
    const uint32_t low  = (uint32_t(1) << drop) - 1u;             // low `drop` bits set
    int total = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u = std::bit_cast<uint32_t>(x[i]);
        total += std::popcount(u & low);
    }
    return total;
}
