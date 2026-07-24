#include "sol.hpp"
#include <bit>
#include <cstdint>

// TODO: keep only the top `keep_bits` of the 23-bit mantissa of each x[i]
// (zero the low 23-keep_bits bits); preserve sign and exponent. In place.
// Reinterpret float<->uint32 with a strict-aliasing-safe mechanism
// (std::bit_cast or std::memcpy). A raw pointer type-pun is UB at -O2.
void quantize_mantissa(float* x, int n, int keep_bits) {
    // your code here
}

// TODO: return the total popcount of the low (23-keep_bits) mantissa bits
// across x[0..n), read with the same strict-aliasing-safe reinterpret.
int count_bits_lost(const float* x, int n, int keep_bits) {
    return 0;  // your code here
}
