#include "sol.hpp"
#include <bit>

// Correct type-punning: copy the bytes with std::bit_cast (never `*(uint32_t*)&x`,
// which is a strict-aliasing violation, and never `(uint32_t)x`, which converts
// the value). std::memcpy would work identically.

uint32_t float_to_bits(float x) {
    return std::bit_cast<uint32_t>(x);
}

float bits_to_float(uint32_t b) {
    return std::bit_cast<float>(b);
}

void scale_pow2_inplace(float* x, int n, int k) {
    for (int i = 0; i < n; i++) {
        uint32_t b    = std::bit_cast<uint32_t>(x[i]);
        uint32_t sign = b & 0x80000000u;          // bit 31
        uint32_t exp  = (b >> 23) & 0xFFu;         // bits 23..30 (biased exponent)
        uint32_t mant = b & 0x007FFFFFu;           // bits 0..22
        exp = (uint32_t)((int)exp + k);            // shift the exponent by k
        uint32_t nb = sign | (exp << 23) | mant;
        x[i] = std::bit_cast<float>(nb);
    }
}
