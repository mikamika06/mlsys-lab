#include "sol.hpp"
#include <arm_neon.h>

std::vector<int32_t> int8_widening_dot_product(const std::vector<int8_t>& A,
                                                 const std::vector<int8_t>& B,
                                                 int M, int N) {
    std::vector<int32_t> res((size_t)M);

    for (int m = 0; m < M; ++m) {
        int32x4_t acc = vdupq_n_s32(0);
        const int8_t* arow = A.data() + (size_t)m * (size_t)N;
        const int8_t* brow = B.data() + (size_t)m * (size_t)N;

        for (int n = 0; n < N; n += 16) {
            int8x16_t va = vld1q_s8(arow + n);
            int8x16_t vb = vld1q_s8(brow + n);
            int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
            int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
            acc = vpadalq_s16(acc, lo);
            acc = vpadalq_s16(acc, hi);
        }

        int32x2_t sum2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
        sum2 = vpadd_s32(sum2, sum2);
        res[(size_t)m] = vget_lane_s32(sum2, 0);
    }

    return res;
}
