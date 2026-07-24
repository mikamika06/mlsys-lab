#include "sol.hpp"

// Reference: the multiply `i * stride` is strength-reduced to an additive
// induction variable `idx`. The result is identical to the multiply form.
long long strided_weighted_sum(const long long* a, int n, int stride) {
    long long acc = 0;
    long long idx = 0;                 // induction variable: replaces i * stride
    for (int i = 0; i < n; i++) {
        acc += (idx + 1) * a[idx];
        idx += stride;                 // add instead of recomputing i * stride
    }
    return acc;
}
