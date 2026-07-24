#pragma once

// Real "SIMD-style" kernels. Each computes the same MATHEMATICAL result as
// a straightforward scalar loop, but the way a real vectorized version
// actually would -- which, for the two reductions below, means the
// additions legitimately happen in a DIFFERENT ORDER than a scalar loop's
// simple left-to-right accumulation. Whether that changes the exact
// floating-point bits is the whole point of this task: main.cpp does not
// ask you to predict anything, it runs your kernel against a scalar
// reference on the same real data and compares the raw bytes with memcmp.

// Elementwise SAXPY: out[i] = a*x[i] + y[i] for i in [0, n). Every output
// element is independent, so ANY processing order gives the exact same
// bits as a scalar loop. (Free to implement this however you like.)
void simdSaxpy(float a, const float* x, const float* y, int n, float* out);

// Sum reduction the way a REAL 4-lane SIMD reduction actually works: keep
// FOUR separate running accumulators (one per lane), add x[i] into
// accumulator (i % 4) as you go, and only combine the four partial sums
// into one value at the very end (e.g. (lane0+lane1) + (lane2+lane3)).
// Do NOT just add everything into a single accumulator in one sequential
// loop -- that is the SCALAR algorithm (main.cpp's reference already does
// that); yours must genuinely reassociate the additions, which for real
// float32 data generally lands on a DIFFERENT last bit than the scalar sum.
float simdFloatSum(const float* x, int n);

// The exact same 4-lane-accumulator reduction, but over int32 data.
// Integer addition is associative modulo 2^32, so lane-reordering is
// expected to land on the exact same bits as a sequential scalar sum
// regardless -- implement it the same way as simdFloatSum, just over ints.
long long simdIntSum(const int* x, int n);

// out[i] = a*x[i] + y[i], computed as a genuinely FUSED multiply-add (ONE
// rounding) via std::fma from <cmath>. Do NOT compute it as a separate
// multiply then add (`a*x[i] + y[i]`, two roundings) -- that is the scalar
// reference's algorithm, and can differ from a true fma by up to 1 ULP.
void simdFma(float a, const float* x, const float* y, int n, float* out);
