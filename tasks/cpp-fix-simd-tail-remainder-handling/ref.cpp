#include "sol.hpp"
#include <arm_neon.h>

long long simd_sum(const int* data, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        acc = vaddq_s32(acc, v);
    }
    long long total = vaddvq_s32(acc);
    // Scalar remainder loop: the last n % 4 elements a whole-chunk-only
    // SIMD loop would otherwise silently drop.
    for (; i < n; i++) {
        total += data[i];
    }
    return total;
}
