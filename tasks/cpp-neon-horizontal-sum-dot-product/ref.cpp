#include "sol.hpp"
#include <arm_neon.h>

float neon_dot(const std::vector<float>& a, const std::vector<float>& b) {
    int n = (int)a.size();
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(&a[(size_t)i]);
        float32x4_t vb0 = vld1q_f32(&b[(size_t)i]);
        acc0 = vfmaq_f32(acc0, va0, vb0);

        float32x4_t va1 = vld1q_f32(&a[(size_t)(i + 4)]);
        float32x4_t vb1 = vld1q_f32(&b[(size_t)(i + 4)]);
        acc1 = vfmaq_f32(acc1, va1, vb1);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[(size_t)i]);
        float32x4_t vb = vld1q_f32(&b[(size_t)i]);
        acc0 = vfmaq_f32(acc0, va, vb);
    }

    float32x4_t acc = vaddq_f32(acc0, acc1);
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    return vget_lane_f32(sum2, 0);
}
