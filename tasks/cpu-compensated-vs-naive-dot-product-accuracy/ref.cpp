#include <cmath>
#include "sol.hpp"

float naive_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

float compensated_dot(const float* a, const float* b, int n) {
    float sum = 0.0f, c = 0.0f;
    for (int i = 0; i < n; i++) {
        float prod = a[i] * b[i];
        float t = sum + prod;
        c += (fabsf(sum) >= fabsf(prod)) ? ((sum - t) + prod) : ((prod - t) + sum);
        sum = t;
    }
    return sum + c;
}
