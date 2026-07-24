#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Real ARM NEON dot product: `a` and `b` have equal length n, n % 4 == 0.
// Compute a . b = sum_i a[i] * b[i] using <arm_neon.h>:
//
//   1. Process elements in groups of 4: load each group with vld1q_f32
//      into a float32x4_t, and multiply-accumulate lane-wise into a
//      float32x4_t accumulator (vmulq_f32 + vaddq_f32, or the fused
//      vfmaq_f32) -- never collapse a group to a scalar inside the loop.
//      You may use two independent float32x4_t accumulators over
//      interleaved groups for instruction-level parallelism, but must
//      merge them (vaddq_f32) before reducing.
//   2. After the loop, perform an EXPLICIT pairwise horizontal reduction
//      of the 4-lane accumulator to a single scalar: combine lanes
//      (0,1) and (2,3) with vadd_f32 on the low/high halves
//      (vget_low_f32 / vget_high_f32), then combine those two partial
//      sums with a second add (e.g. vpadd_f32), and extract the final
//      scalar with vget_lane_f32. Do not just index into the vector's
//      lanes and add them as four separate plain scalars -- write the
//      lane-pairwise reduction.
// ---------------------------------------------------------------------------
float neon_dot(const std::vector<float>& a, const std::vector<float>& b);
