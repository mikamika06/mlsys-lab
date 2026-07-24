#include "sol.hpp"
#include <arm_neon.h>

// BUG: only processes whole 4-wide chunks; any remainder (n % 4 elements)
// at the end of the array is silently dropped.
long long simd_sum(const int* data, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t v = vld1q_s32(data + i);
        acc = vaddq_s32(acc, v);
    }
    long long total = vaddvq_s32(acc);
    return total;
}
