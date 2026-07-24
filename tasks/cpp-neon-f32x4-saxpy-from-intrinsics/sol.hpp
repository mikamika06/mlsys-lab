#pragma once
// SAXPY (single-precision a*x + y), in place, four lanes at a time.
//
// Compute y[i] = a * x[i] + y[i] for i in [0, n).
// n is a multiple of 4 so the whole range fits exact 128-bit NEON blocks.
// Implement this with ARM NEON f32x4 intrinsics from <arm_neon.h>:
//   vdupq_n_f32, vld1q_f32, vmlaq_f32 (or vfmaq_f32), vst1q_f32.
void saxpy_neon(float a, const float* x, float* y, int n);
