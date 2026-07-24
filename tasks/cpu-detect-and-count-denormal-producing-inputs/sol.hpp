#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Count how many of arr[0..n) are SUBNORMAL (a.k.a. denormal) IEEE-754
// floats: nonzero values whose magnitude is strictly less than FLT_MIN
// (the smallest positive NORMAL float, ~1.1754944e-38).
//
//   - +0.0 / -0.0            -> NOT counted (zero is its own class).
//   - |x| == FLT_MIN exactly -> NOT counted (that's the smallest NORMAL).
//   - 0 < |x| < FLT_MIN      -> counted (this is the subnormal range).
//   - +-infinity, NaN        -> NOT counted.
// ============================================================================
int count_denormals(const float* arr, int n);
