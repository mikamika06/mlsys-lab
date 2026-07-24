#pragma once
#include <cstdint>

// Reduced-precision (mantissa-truncation) quantization of IEEE-754 float32.
// For each x[i], keep the top `keep_bits` (0..23) bits of the 23-bit mantissa
// and zero the remaining low bits; the sign and 8-bit exponent are preserved.
// Done in place. This is the core of storing weights in a reduced-precision
// float format (e.g. bfloat16 keeps only the top 7 mantissa bits).
//
// Implement the float<->uint32 reinterpret with a STRICT-ALIASING-SAFE
// mechanism (std::bit_cast<uint32_t>/<float> or std::memcpy). A raw pointer
// type-pun such as `*(uint32_t*)&x[i]` is undefined behavior under the C++
// strict-aliasing rule and may be silently miscompiled at -O2.
void quantize_mantissa(float* x, int n, int keep_bits);

// Return the total number of mantissa bits that quantize_mantissa(., keep_bits)
// discards across x[0..n): the popcount of the low (23 - keep_bits) mantissa
// bits, summed over every element. Read the float bits with the same
// strict-aliasing-safe reinterpret.
int count_bits_lost(const float* x, int n, int keep_bits);
