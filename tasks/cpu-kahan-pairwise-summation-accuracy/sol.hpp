#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Kahan (compensated) summation of arr[0..n) -- all in float32, no double
// or long double allowed anywhere. Track a running compensation term `c`
// that captures the low-order bits lost to rounding on each addition, and
// feed it back in on the next step:
//
//     sum = 0, c = 0
//     for each x in arr:
//         y   = x - c        // subtract off the running correction
//         t   = sum + y      // this addition is where new rounding happens
//         c   = (t - sum) - y  // recover the bits that got rounded away
//         sum = t
//     return sum
//
// A naive `sum += arr[i]` loop uses the SAME float32 precision but has no
// compensation term, so it can silently lose entire increments when the
// running total's magnitude is much larger than the increment being added
// (the increment falls below the accumulator's ULP and rounds to nothing).
// ============================================================================
float kahan_sum(const float* arr, int n);
