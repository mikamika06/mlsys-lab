#include "sol.hpp"

// TODO: compute row-wise int8 dot products into int32 accumulators using
// NEON widening intrinsics (vld1q_s8 / vmull_s8 / vpadalq_s16). See
// sol.hpp for the exact contract.
std::vector<int32_t> int8_widening_dot_product(const std::vector<int8_t>& A,
                                                 const std::vector<int8_t>& B,
                                                 int M, int N) {
    (void)A;
    (void)B;
    (void)M;
    (void)N;
    return {};
}
