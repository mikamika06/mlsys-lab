#pragma once
#include <cstdint>

// ============================================================================
// LEARNER implements both functions in solve.cpp, using ONLY arithmetic /
// bitwise mask tricks -- no `if`, `?:`, `std::min`, `std::max`, or any other
// construct that branches on a *data* value. (Fixed-trip-count loops such as
// `for (int i = 0; i < n; ++i)` are fine: their trip count never depends on
// the array contents, so the CPU predicts them perfectly -- that kind of
// branch is not the lesson here. The lesson is the *data-dependent* branch
// that used to live inside the loop body.)
// ============================================================================

// select_branchless: out[i] = cond[i] ? a[i] : b[i], for i in [0, n).
// Build a full-width mask (all-1 bits if cond[i] != 0, all-0 bits
// otherwise) and combine a[i]/b[i] through it -- do not test cond[i] with a
// per-element branch.
void select_branchless(const int32_t* cond, const int32_t* a, const int32_t* b,
                        int32_t* out, int n);

// clamp_branchless: out[i] = clamp(x[i], lo, hi), for i in [0, n).
// Build this from the sign-bit branchless min/max identity
// (min(a,b) = b + ((a-b) & ((a-b) >> 31)), and max similarly), never from
// std::min/std::max or a per-element compare-and-branch.
void clamp_branchless(const int32_t* x, int32_t lo, int32_t hi,
                       int32_t* out, int n);
