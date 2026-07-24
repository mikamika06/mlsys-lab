#pragma once
#include <vector>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// log_sum_exp: given a non-empty vector x, compute
//     log( sum_i exp(x[i]) )
// EXACTLY (to double precision), even when the x[i] span hundreds of orders
// of magnitude -- large enough that exp(x[i]) alone would overflow to +inf,
// or negative enough that exp(x[i]) alone would underflow to 0.
//
// The naive one-line translation of the formula computes exp() first and
// log() second, and both intermediate steps can lose all the information
// the final answer needed. The fix is the standard shift-by-the-max trick:
//     m = max_i x[i]
//     log_sum_exp(x) = m + log( sum_i exp(x[i] - m) )
// Every shifted exponent x[i] - m is <= 0, so every exp() term is in
// (0, 1] -- never overflows, and the largest term is always exactly 1, so
// the sum is never all-zero underflow either.
// ============================================================================
double log_sum_exp(const std::vector<double>& x);
