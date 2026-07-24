#pragma once

// Plain float32 dot product: straight left-to-right accumulation,
// sum += a[i] * b[i], no rounding compensation. This is what everyone
// writes first, and it silently drops low-order bits whenever the
// running sum's magnitude is much larger than the next product.
float naive_dot(const float* a, const float* b, int n);

// Kahan-Neumaier compensated dot product: track a running compensation
// term `c` that captures the bits each addition rounds away, and feed
// it back in on the NEXT addition instead of discarding it. Unlike
// plain Kahan summation, Neumaier's variant picks up the low-order bits
// correctly whether the running sum or the new term is larger in
// magnitude, so it works even when a single small term outweighs the
// accumulated sum (as it does here early in the loop).
//
// For each i, with running float sum (init 0) and float c (init 0):
//   float prod = a[i] * b[i];
//   float t = sum + prod;
//   c += (fabsf(sum) >= fabsf(prod)) ? ((sum - t) + prod)
//                                     : ((prod - t) + sum);
//   sum = t;
// After the loop, the corrected result is sum + c.
float compensated_dot(const float* a, const float* b, int n);
