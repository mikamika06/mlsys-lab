#include "sol.hpp"

// TODO: for each w in [1,max_w], count distinct addrs[i]/line_bytes over
// the LAST w elements (addrs[n-w..n-1]) into curve_out[w-1]; return the
// smallest w where curve_out[w-1] equals curve_out[max_w-1]. See sol.hpp.
int plateau_index(const long* addrs, int n, int max_w, int line_bytes, int* curve_out) {
    (void)addrs; (void)n; (void)line_bytes;
    // your code here
    for (int w = 0; w < max_w; w++) curve_out[w] = 0;
    return 1;
}
