#include <arm_neon.h>
#include "sol.hpp"

// Reference: real ARM NEON vzipq_f32 shuffles compose the transpose.
void transpose4x4(const float* in, float* out) {
    float32x4_t r0 = vld1q_f32(in + 0);
    float32x4_t r1 = vld1q_f32(in + 4);
    float32x4_t r2 = vld1q_f32(in + 8);
    float32x4_t r3 = vld1q_f32(in + 12);

    // Stage 1: zip rows 0&2 and rows 1&3.
    float32x4x2_t t0 = vzipq_f32(r0, r2);   // t0.val[0]=[r0_0,r2_0,r0_1,r2_1], t0.val[1]=[r0_2,r2_2,r0_3,r2_3]
    float32x4x2_t t1 = vzipq_f32(r1, r3);   // t1.val[0]=[r1_0,r3_0,r1_1,r3_1], t1.val[1]=[r1_2,r3_2,r1_3,r3_3]

    // Stage 2: zip the stage-1 pairs together lane-by-lane -> transposed rows.
    float32x4x2_t res_lo = vzipq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t res_hi = vzipq_f32(t0.val[1], t1.val[1]);

    vst1q_f32(out + 0,  res_lo.val[0]);
    vst1q_f32(out + 4,  res_lo.val[1]);
    vst1q_f32(out + 8,  res_hi.val[0]);
    vst1q_f32(out + 12, res_hi.val[1]);
}
