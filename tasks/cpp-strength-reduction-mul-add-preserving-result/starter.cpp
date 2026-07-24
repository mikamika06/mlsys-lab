#include "sol.hpp"

// TODO: implement strided_weighted_sum using an additive induction variable
// (idx += stride) in place of the i * stride multiply, and return the same acc
// the reference would. Right now it returns 0 for every fixture, so it fails.
long long strided_weighted_sum(const long long* a, int n, int stride) {
    (void)a; (void)n; (void)stride;
    return 0;  // your code here
}
