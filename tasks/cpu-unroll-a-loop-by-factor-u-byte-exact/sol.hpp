#pragma once

// LEARNER IMPLEMENTS.
//
// Compute, for every i in [0, n): y[i] = y[i] + a * x[i] -- in place,
// writing the result back into `y`. `n` is guaranteed to be an exact
// multiple of `U` (no remainder handling needed).
//
// Write the loop UNROLLED by a factor of `U`: the outer loop must
// advance `U` elements at a time (`for (int b = 0; b < n; b += U)`),
// and each pass through its body computes all `U` of that block's
// elements before moving to the next block. Element `k` within block
// `b` (for `k` in `[0, U)`) is at index `b + k`. `U` is a *runtime*
// parameter here (a real hand-unrolled kernel would hardcode a
// specific U and write its body as U separate straight-line
// statements instead of an inner loop) -- getting the block-index
// arithmetic right for every U is the point of this exercise.
void axpy_unrolled(int n, int U, float a, const float* x, float* y);
