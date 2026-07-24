#include "sol.hpp"

// TODO: compute a . b using NEON float32x4_t multiply-accumulate over
// groups of 4, then an explicit pairwise horizontal reduction. See
// sol.hpp for the exact contract.
float neon_dot(const std::vector<float>& a, const std::vector<float>& b) {
    (void)a;
    (void)b;
    return 0.0f;
}
