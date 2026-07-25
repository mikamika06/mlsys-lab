#include "sol.hpp"

// VALID but suboptimal: a left-to-right sequential sum. Every add depends
// on the one before it, so the critical path grows to n-1 no matter how
// many independent adders the CPU has.
float parallel_sum(const float* values, int n) {
    if (n == 0) return 0.0f;
    float acc = values[0];
    int d = 0;
    for (int i = 1; i < n; ++i) {
        acc = acc + values[i];
        d = record_add(d, 0);
    }
    return acc;
}
