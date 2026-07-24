#include "sol.hpp"
#include <arm_neon.h>

// Reference: NEON f32x4 SAXPY, four lanes per iteration.
// y[i] = a * x[i] + y[i].
void saxpy_neon(float a, const float* x, float* y, int n) {
    float32x4_t va = vdupq_n_f32(a);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vy = vld1q_f32(y + i);
        vy = vmlaq_f32(vy, va, vx);   // vy + va*vx, elementwise
        vst1q_f32(y + i, vy);
    }
    for (; i < n; i++) y[i] = a * x[i] + y[i];  // scalar tail (n need not divide 4)
}
