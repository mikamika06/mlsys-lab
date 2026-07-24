// Fixed driver: a fixed table of (a, b, c) float triples, each run
// through both fma_result and naive_result. No timing, no randomness --
// every input is a hardcoded float literal, so behavior is fully
// determined by IEEE-754 float arithmetic.
#include "sol.hpp"
#include <cstdio>

namespace {
struct Case { float a, b, c; };

// The first five triples are engineered so that a*b nearly cancels with
// c (the classic setting where double rounding shows up: a small result
// built from the subtraction of two much larger nearly-equal
// magnitudes). The last triple uses exactly-representable integers, so
// the product and sum are exact and there is nothing to round -- FMA and
// the naive path agree.
const Case CASES[] = {
    {1.8944242f,    1.83523202f,  -3.47798467f},
    {1.29811692f,   1.39331698f,  -1.80703771f},
    {1.83851552f,   1.9419601f,   -3.57370281f},
    {1.48461676f,   1.44064152f,  -2.13747525f},
    {0.512582421f,  0.937253594f, -0.480797857f},
    {2.0f,          3.0f,          4.0f},
};
const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);
} // namespace

int main() {
    for (int i = 0; i < NUM_CASES; i++) {
        const Case& t = CASES[i];
        float fma_r = fma_result(t.a, t.b, t.c);
        float naive_r = naive_result(t.a, t.b, t.c);
        printf("case%d: fma=%.9g naive=%.9g diff=%.9g\n", i, (double)fma_r, (double)naive_r,
               (double)fma_r - (double)naive_r);
    }
    return 0;
}
