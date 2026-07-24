#include <cmath>
#include <cstddef>
#include "sol.hpp"

namespace {

float forward_sum(const std::vector<float>& x) {
    float s = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) s = s + x[i];
    return s;
}

float reverse_sum(const std::vector<float>& x) {
    float s = 0.0f;
    for (size_t i = x.size(); i-- > 0;) s = s + x[i];
    return s;
}

float pairwise_sum(const std::vector<float>& x, size_t lo, size_t hi) {
    size_t len = hi - lo;
    if (len == 0) return 0.0f;
    if (len == 1) return x[lo];
    size_t mid = lo + len / 2;
    return pairwise_sum(x, lo, mid) + pairwise_sum(x, mid, hi);
}

float kahan_sum(const std::vector<float>& x) {
    float sum = 0.0f, c = 0.0f;
    for (float v : x) {
        float y = v - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

}  // namespace

std::vector<double> sum_ordering_rel_errors(const std::vector<float>& x) {
    double ref64 = 0.0;
    for (float v : x) ref64 += static_cast<double>(v);

    float s_fwd = forward_sum(x);
    float s_rev = reverse_sum(x);
    float s_pair = pairwise_sum(x, 0, x.size());
    float s_kahan = kahan_sum(x);

    std::vector<double> out(4);
    out[0] = std::fabs(static_cast<double>(s_fwd) - ref64) / std::fabs(ref64);
    out[1] = std::fabs(static_cast<double>(s_rev) - ref64) / std::fabs(ref64);
    out[2] = std::fabs(static_cast<double>(s_pair) - ref64) / std::fabs(ref64);
    out[3] = std::fabs(static_cast<double>(s_kahan) - ref64) / std::fabs(ref64);
    return out;
}
