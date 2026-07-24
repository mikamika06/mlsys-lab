#pragma once
// Reduction with K independent accumulators (breaks the serial FP add chain).
//
// A naive sum accumulates into ONE variable, so every add must wait for the
// previous one to retire (a loop-carried dependency the length of the array).
// Splitting the work across K accumulators removes that chain and lets the CPU
// overlap the adds — while producing the same mathematical sum.
//
// Use the following FIXED strided assignment: accumulator j (0 <= j < K)
// receives every K-th element starting at index j, i.e.
//
//     partial[j] = sum over all i in [0, n) with (i % K == j) of x[i]
//
// Write the K partial sums into partial[0..K) and return their total, which
// equals the full reduction sum of x[0..n).
double reduce_multi_acc(const double* x, int n, double* partial, int K);
