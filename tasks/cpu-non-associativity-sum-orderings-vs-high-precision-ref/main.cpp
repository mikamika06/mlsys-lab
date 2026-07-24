#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Deterministic fixture (no rand()/time()): one large value
// (2^25) followed by 100000 copies of 1.0f -- chosen so that at float32
// precision, adding 1.0 to something near 2^25 (ULP = 4 there) rounds away
// to nothing, but the true sum is exactly representable in both float32
// and float64.

int main() {
    constexpr int kOnes = 100000;
    std::vector<float> x;
    x.reserve(kOnes + 1);
    x.push_back(33554432.0f);  // 2^25
    for (int i = 0; i < kOnes; ++i) x.push_back(1.0f);

    std::vector<double> errs = sum_ordering_rel_errors(x);

    printf("n=%zu\n", x.size());
    const char* names[4] = {"forward", "reverse", "pairwise", "kahan"};
    for (int i = 0; i < 4 && i < static_cast<int>(errs.size()); ++i) {
        printf("%s rel_err=%.10e\n", names[i], errs[i]);
    }
    return 0;
}
