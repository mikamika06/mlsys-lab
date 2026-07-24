#include "sol.hpp"
#include <cmath>

void simdSaxpy(float a, const float* x, const float* y, int n, float* out) {
    // process in blocks of 4 "lanes" -- elementwise, so order doesn't matter
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        for (int lane = 0; lane < 4; lane++) out[i + lane] = a * x[i + lane] + y[i + lane];
    }
    for (; i < n; i++) out[i] = a * x[i] + y[i];
}

float simdFloatSum(const float* x, int n) {
    float lanes[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int i = 0;
    for (; i < n; i++) lanes[i % 4] += x[i];
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}

long long simdIntSum(const int* x, int n) {
    long long lanes[4] = {0, 0, 0, 0};
    int i = 0;
    for (; i < n; i++) lanes[i % 4] += x[i];
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}

void simdFma(float a, const float* x, const float* y, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = std::fma(a, x[i], y[i]);
}
