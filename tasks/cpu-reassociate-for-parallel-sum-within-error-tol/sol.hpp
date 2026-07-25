#pragma once

// ============================================================================
// Dependency-depth model for a chain of floating-point adds (FIXED — do
// not modify; defined in main.cpp). Every raw input value has depth 0
// (nothing had to complete before it's available). Call record_add(a, b)
// exactly once per '+' your summation performs, passing the DEPTH of each
// of its two operands (0 for a value straight out of the input array you
// have not added yet, or the depth previously returned by record_add for
// a partial sum you are now combining further). It returns the new
// value's depth: 1 + max(depth_a, depth_b) -- how many dependent adds,
// back-to-back, had to happen in sequence to produce it.
// ============================================================================
void reset_critical_path();
int record_add(int depth_a, int depth_b);
int critical_path_depth(); // the largest depth returned by record_add so far

// ============================================================================
// Sum all n floats in `values` and return the sum. `n` is a power of two.
// A left-to-right sequential sum has a critical path of n-1 dependent
// adds -- each one has to wait for the one before it, no matter how many
// independent execution units the CPU has. Reassociate the summation
// (pairwise/tree reduction) so independent partial sums can be computed
// in parallel, keeping the critical path down near log2(n). Report every
// '+' you perform through record_add, mirroring your actual addition
// tree.
// ============================================================================
float parallel_sum(const float* values, int n);
