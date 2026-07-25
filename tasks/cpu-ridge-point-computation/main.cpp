#include <cstdio>
#include "sol.hpp"

// FIXED driver: 5 real-ish device specs (peak FLOP/s, peak bytes/s)
// spanning several orders of magnitude, so a formula bug (operands
// swapped, wrong which-array-is-which) produces numbers off by orders
// of magnitude rather than coincidentally matching by luck.
int main() {
    static const double PEAK_FLOPS[5] = {16e12, 9.7e12, 312e12, 1e12, 125e12};
    static const double PEAK_BW[5]    = {2e12, 900e9, 2.039e12, 500e9, 3.35e12};

    for (int i = 0; i < 5; i++) {
        double r = ridge_point(PEAK_FLOPS[i], PEAK_BW[i]);
        printf("ridge[%d]=%.10g\n", i, r);
    }
    return 0;
}
