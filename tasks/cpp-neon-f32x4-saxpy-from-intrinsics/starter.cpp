#include "sol.hpp"
#include <arm_neon.h>

// TODO: implement SAXPY y[i] = a*x[i] + y[i] with NEON f32x4 intrinsics.
// Broadcast a with vdupq_n_f32, load x/y with vld1q_f32, fuse with
// vmlaq_f32 (or vfmaq_f32), store back with vst1q_f32; scalar tail for leftovers.
void saxpy_neon(float a, const float* x, float* y, int n) {
    (void)a; (void)x; (void)y; (void)n;
    // your code here
}
