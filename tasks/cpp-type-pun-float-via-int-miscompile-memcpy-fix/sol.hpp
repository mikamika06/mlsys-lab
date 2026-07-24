#pragma once
#include <cstdint>

// --- Type-punning floats and their IEEE-754 bit patterns ---
//
// A 32-bit `float` and a 32-bit unsigned integer occupy the same four bytes,
// but they are DIFFERENT TYPES. To view the bytes of one as the other you must
// COPY the bytes (std::bit_cast<...> or std::memcpy).
//
// Two traps this task is about:
//   * Reading a float object through an `int*` / `unsigned*`
//     (e.g. `*(uint32_t*)&x`) is a strict-aliasing violation: undefined
//     behavior that clang++ is free to MISCOMPILE at -O2.
//   * A numeric cast (`(uint32_t)x`) is different again — it rounds the
//     *value*, throwing the bit pattern away.
// The correct, portable tool for reinterpreting the bytes is std::bit_cast
// (C++20) or std::memcpy.

// Return the raw 32-bit IEEE-754 bit pattern of x (its bytes, not its value).
uint32_t float_to_bits(float x);

// Reconstruct the float whose raw 32-bit IEEE-754 bit pattern is b.
float bits_to_float(uint32_t b);

// In place: multiply each x[i] by 2^k by adding k to the biased exponent field
// of its IEEE-754 bit pattern (an exact power-of-two scale for the given normal
// inputs, whose results also stay normal). n elements; k may be negative.
void scale_pow2_inplace(float* x, int n, int k);
