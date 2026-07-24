#include "sol.hpp"

void select_branchless(const int32_t* cond, const int32_t* a, const int32_t* b,
                        int32_t* out, int n) {
    // your code here
    (void)cond;
    (void)a;
    (void)b;
    for (int i = 0; i < n; ++i) out[i] = 0;
}

void clamp_branchless(const int32_t* x, int32_t lo, int32_t hi,
                       int32_t* out, int n) {
    // your code here
    (void)x;
    (void)lo;
    (void)hi;
    for (int i = 0; i < n; ++i) out[i] = 0;
}
