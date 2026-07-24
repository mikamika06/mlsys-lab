#include "sol.hpp"

// TODO: these implementations are WRONG. A numeric cast rounds the *value* and
// throws the bit pattern away; it does NOT reinterpret the bytes. Reinterpret
// the bytes with std::bit_cast<...> (or std::memcpy) instead — and never with
// `*(uint32_t*)&x`, which is a strict-aliasing violation the compiler is allowed
// to miscompile at -O2.

uint32_t float_to_bits(float x) {
    return (uint32_t)x;          // BUG: rounds the value, loses the bit pattern
}

float bits_to_float(uint32_t b) {
    return (float)b;             // BUG: converts the integer value, not the bits
}

void scale_pow2_inplace(float* x, int n, int k) {
    // TODO: for each x[i], add k to its biased exponent field via the bit
    //       pattern, then write the reinterpreted float back.
    (void)x; (void)n; (void)k;   // does nothing yet
}
