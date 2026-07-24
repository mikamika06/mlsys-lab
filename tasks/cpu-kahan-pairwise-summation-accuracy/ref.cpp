#include "sol.hpp"

float kahan_sum(const float* arr, int n) {
    float sum = 0.0f, c = 0.0f;
    for (int i = 0; i < n; i++) {
        float y = arr[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
