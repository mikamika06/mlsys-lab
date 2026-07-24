#include <cmath>
#include "sol.hpp"

int count_denormals(const float* arr, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (std::fpclassify(arr[i]) == FP_SUBNORMAL) count++;
    }
    return count;
}
