#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver, two adversarial fixtures. A naive `sum += x` loop is
// computed here (independently of the graded function) purely so its
// value shows up alongside for context -- it does not affect grading.
namespace {
float naive_sum(const float* arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum;
}
}  // namespace

int main() {
    // Fixture 1: a 1e8-magnitude base (ULP ~8 there) followed by 100000
    // increments of 1.0f -- each individual increment is far below the
    // accumulator's ULP, so naive summation drops them entirely.
    std::vector<float> f1;
    f1.push_back(1e8f);
    for (int i = 0; i < 100000; i++) f1.push_back(1.0f);

    // Fixture 2: one million copies of 0.1f (not exactly representable in
    // binary) -- naive left-to-right summation drifts as rounding error
    // compounds across a million additions.
    std::vector<float> f2(1000000, 0.1f);

    float k1 = kahan_sum(f1.data(), (int)f1.size());
    float k2 = kahan_sum(f2.data(), (int)f2.size());
    float n1 = naive_sum(f1.data(), (int)f1.size());
    float n2 = naive_sum(f2.data(), (int)f2.size());

    printf("kahan1=%.6f naive1=%.6f\n", k1, n1);
    printf("kahan2=%.6f naive2=%.6f\n", k2, n2);
    return 0;
}
