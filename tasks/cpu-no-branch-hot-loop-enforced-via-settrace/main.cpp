// Fixed driver: a hot loop that calls clamp_branchless once per fixed
// test case (values inside range, below lo, above hi, exactly on each
// boundary, and a degenerate lo==hi==x case), then prints every result
// plus the total number of branchy_min/branchy_max calls made across
// the whole run. No timing, no randomness.
#include "sol.hpp"
#include <cmath>
#include <cstdio>

namespace {
long g_branch_calls = 0;
}

float branchless_min(Guarded a, Guarded b) { return std::fminf(a.v, b.v); }
float branchless_max(Guarded a, Guarded b) { return std::fmaxf(a.v, b.v); }

float branchy_min(Guarded a, Guarded b) {
    g_branch_calls++;
    if (a.v < b.v) return a.v;
    return b.v;
}
float branchy_max(Guarded a, Guarded b) {
    g_branch_calls++;
    if (a.v > b.v) return a.v;
    return b.v;
}

int main() {
    struct Case {
        float x, lo, hi;
    };
    static const Case CASES[] = {
        {5.0f, 0.0f, 10.0f},        // inside range
        {-3.0f, 0.0f, 10.0f},       // below lo
        {15.0f, 0.0f, 10.0f},       // above hi
        {0.0f, 0.0f, 10.0f},        // exactly lo
        {10.0f, 0.0f, 10.0f},       // exactly hi
        {-1e6f, -100.0f, 100.0f},   // far below
        {1e6f, -100.0f, 100.0f},    // far above
        {2.5f, 2.5f, 2.5f},         // degenerate lo == hi == x
    };
    const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);

    for (int i = 0; i < NUM_CASES; i++) {
        const Case& c = CASES[i];
        float r = clamp_branchless(Guarded::wrap(c.x), Guarded::wrap(c.lo), Guarded::wrap(c.hi));
        printf("case%d: clamp(%.6g,%.6g,%.6g)=%.6g\n", i, (double)c.x, (double)c.lo, (double)c.hi,
               (double)r);
    }
    printf("branchy_calls=%ld\n", g_branch_calls);
    return 0;
}
