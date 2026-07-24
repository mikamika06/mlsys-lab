#include "sol.hpp"

float clamp_branchless(Guarded x, Guarded lo, Guarded hi) {
    float below_hi = branchless_min(x, hi);
    return branchless_max(Guarded::wrap(below_hi), lo);
}
