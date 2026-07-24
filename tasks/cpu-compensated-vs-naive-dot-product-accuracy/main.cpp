#include <cstdio>
#include <cmath>
#include "sol.hpp"

// Deterministic xorshift PRNG (harness code, fixed seed -- no rand(),
// no clock).
namespace {
unsigned long g_state = 88172645463325252UL;
unsigned long xorshift() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}
float rand_unit() { return (float)((xorshift() % 1000000UL) / 1000000.0); }
}  // namespace

// FIXED driver. Builds two length-N float32 vectors whose products
// alternate between "big" terms (~1e8 magnitude, one in every 5) and
// "small" terms (~1e-8 magnitude, the rest) -- every big term dwarfs the
// small ones enough that plain float32 addition rounds a small term
// away completely once the running sum has absorbed a big one. Computes
// a double-precision reference dot product directly (the harness's own
// ground truth, not something either candidate function computes), then
// compares both candidate results against it.
int main() {
    const int N = 200000;
    static float a[N], b[N];
    for (int i = 0; i < N; i++) {
        if (i % 5 == 0) {
            a[i] = 1.0e4f + rand_unit() * 1.0e4f;
            b[i] = 1.0e4f + rand_unit() * 1.0e4f;
        } else {
            a[i] = (rand_unit() - 0.5f) * 2.0e-4f;
            b[i] = (rand_unit() - 0.5f) * 2.0e-4f;
        }
    }

    double ref = 0.0;
    for (int i = 0; i < N; i++) ref += (double)a[i] * (double)b[i];

    float naive = naive_dot(a, b, N);
    float comp = compensated_dot(a, b, N);

    double naive_rel = fabs((double)naive - ref) / fabs(ref);
    double comp_rel = fabs((double)comp - ref) / fabs(ref);

    printf("ref=%.9e naive=%.9e naive_rel=%.6e comp=%.9e comp_rel=%.6e\n",
           ref, (double)naive, naive_rel, (double)comp, comp_rel);
    return 0;
}
