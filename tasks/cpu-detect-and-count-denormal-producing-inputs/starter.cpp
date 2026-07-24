#include "sol.hpp"

// TODO: for each arr[i], it's subnormal iff it's nonzero AND its magnitude
// is strictly less than FLT_MIN (equivalently: std::fpclassify(arr[i]) ==
// FP_SUBNORMAL). Count how many satisfy that.
int count_denormals(const float* arr, int n) {
    (void)arr; (void)n;
    // your code here
    return 0;
}
