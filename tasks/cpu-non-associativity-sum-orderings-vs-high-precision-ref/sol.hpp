#pragma once
#include <vector>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// sum_ordering_rel_errors: given `x` (single-precision values), sum it FOUR
// different ways, every accumulation step done in float32 arithmetic (the
// running total is a `float`, not a `double`):
//   0. forward   -- left to right, index 0 .. n-1
//   1. reverse   -- right to left, index n-1 .. 0
//   2. pairwise  -- recursive divide-and-conquer: split [lo, hi) at the
//      midpoint, sum each half recursively, add the two half-sums
//   3. kahan     -- Kahan compensated summation (running compensation term
//      that captures the low-order bits float32 addition would otherwise
//      drop, and feeds them back in on the next step)
//
// Also compute a high-precision reference: the SAME values summed
// left-to-right with a float64 (`double`) accumulator.
//
// Return a 4-element vector of RELATIVE errors, one per ordering above, each
// computed as |sum32 - ref64| / |ref64| (using the float32 result promoted
// to double for the comparison, and the float64 reference as-is).
// ============================================================================
std::vector<double> sum_ordering_rel_errors(const std::vector<float>& x);
