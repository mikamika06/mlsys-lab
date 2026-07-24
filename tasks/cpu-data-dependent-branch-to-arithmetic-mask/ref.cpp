#include <cstdint>
#include "sol.hpp"

void select_branchless(const int32_t* cond, const int32_t* a, const int32_t* b,
                        int32_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        uint32_t mask = static_cast<uint32_t>(-(cond[i] != 0));  // all-1s or all-0s
        uint32_t ua = static_cast<uint32_t>(a[i]);
        uint32_t ub = static_cast<uint32_t>(b[i]);
        out[i] = static_cast<int32_t>((ua & mask) | (ub & ~mask));
    }
}

void clamp_branchless(const int32_t* x, int32_t lo, int32_t hi,
                       int32_t* out, int n) {
    for (int i = 0; i < n; ++i) {
        // max(x[i], lo) via the sign-bit identity: max(a,b) = a - ((a-b) & ((a-b)>>31))
        int32_t d1 = x[i] - lo;
        int32_t hi_of_lo = x[i] - (d1 & (d1 >> 31));
        // min(hi_of_lo, hi) via: min(a,b) = b + ((a-b) & ((a-b)>>31))
        int32_t d2 = hi_of_lo - hi;
        out[i] = hi + (d2 & (d2 >> 31));
    }
}
