#pragma once
// Strength reduction (mul -> add), result-preserving.
//
// Compute, over an array `a` of length >= (n-1)*stride + 1:
//
//     acc = 0
//     for i in [0, n):
//         idx = i * stride              // index into a
//         acc += (idx + 1) * a[idx]     // weight each element by (idx + 1)
//     return acc
//
// The reference eliminates the per-iteration `i * stride` multiply by carrying an
// additive INDUCTION VARIABLE (idx starts at 0 and does idx += stride each step),
// then reuses that same idx for both the subscript and the (idx + 1) weight.
//
// Your implementation must return the EXACT same acc for every fixture — the
// strength-reduced form has to preserve the result bit-for-bit.
//
// Preconditions: n >= 0, stride >= 1, and (n-1)*stride is a valid index into a.
long long strided_weighted_sum(const long long* a, int n, int stride);
