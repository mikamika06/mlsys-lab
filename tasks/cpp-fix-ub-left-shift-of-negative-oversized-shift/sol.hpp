#pragma once

// Fix the undefined behavior in a batch left-shift kernel.
//
// For each i in [0, n): compute the SAFE, well-defined equivalent of
// `(long)(values[i] << shift_amounts[i])` and store it in results[i]:
//
//   1. Reinterpret values[i] as `unsigned int` (bitcast, not a value
//      conversion -- this is what a negative signed value's bit pattern
//      looks like unsigned).
//   2. Clamp shift_amounts[i] to [0, 31] by taking it modulo 32 as an
//      unsigned quantity (shift_amounts[i] is never negative in the test
//      data, but may be >= 32).
//   3. Left-shift the unsigned value by that clamped amount -- unsigned
//      shift is always well-defined, it simply drops bits off the top.
//   4. Reinterpret the 32-bit unsigned result back as a signed `int`
//      (two's-complement bitcast), then store it into the 64-bit `long`
//      result, which sign-extends it.
void process_shifts(const int* values, const int* shift_amounts, long* results, int n);
