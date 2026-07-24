#include "sol.hpp"

// BUG: every "SIMD" kernel here is really just the scalar algorithm again
// -- a single sequential accumulator instead of 4 lanes, and a plain
// `a*x[i]+y[i]` instead of a genuine std::fma. That happens to still be
// correct for the elementwise/associative cases, but it means the
// reduction and FMA kernels never actually diverge from a scalar
// implementation the way a real vectorized one would.
void simdSaxpy(float a, const float* x, const float* y, int n, float* out) {
    for (int i = 0; i < n; i++) out[i] = a * x[i] + y[i];
}

float simdFloatSum(const float* x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) acc += x[i];
    return acc;
}

long long simdIntSum(const int* x, int n) {
    long long acc = 0;
    for (int i = 0; i < n; i++) acc += x[i];
    return acc;
}

void simdFma(float a, const float* x, const float* y, int n, float* out) {
    for (int i = 0; i < n; i++) {
        volatile float prod = a * x[i];  // exactly the scalar algorithm: two roundings
        out[i] = prod + y[i];
    }
}
