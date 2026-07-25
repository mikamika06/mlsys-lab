// Fixed driver: for each of 5 fixed bias values p, runs an actual
// deterministic simulation of the 2-bit saturating counter over a long
// i.i.d. Bernoulli(p) trace (xorshift32 PRNG, fixed seed -- no
// wall-clock, no system rand()), then prints the candidate's derived
// theoretical rate alongside that independently-simulated empirical
// rate, so the two can be compared by eye.
#include "sol.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {
uint32_t g_state = 2463534242u; // fixed nonzero seed

uint32_t xorshift32() {
    uint32_t x = g_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_state = x;
    return x;
}

// Uniform double in [0, 1) built from 24 pseudo-random bits.
double next_uniform() {
    return static_cast<double>(xorshift32() & 0xFFFFFFu) / 16777216.0;
}

// Independent HARNESS simulation (not learner code): actually run the
// 2-bit counter over `n` i.i.d. Bernoulli(p) branch outcomes and return
// the empirical mispredict rate.
double simulate_mispredict_rate(double p, long n) {
    int state = 1; // start at Weakly Not-taken, arbitrary
    long mispredicts = 0;
    for (long i = 0; i < n; i++) {
        bool taken = next_uniform() < p;
        bool predict_taken = state >= 2;
        if (predict_taken != taken) mispredicts++;
        if (taken) {
            state = state < 3 ? state + 1 : 3;
        } else {
            state = state > 0 ? state - 1 : 0;
        }
    }
    return static_cast<double>(mispredicts) / static_cast<double>(n);
}
} // namespace

int main() {
    static const double P_VALUES[] = {0.1, 0.3, 0.5, 0.7, 0.9};
    const int NUM_P = sizeof(P_VALUES) / sizeof(P_VALUES[0]);
    const long N = 200000;

    for (int i = 0; i < NUM_P; i++) {
        double p = P_VALUES[i];
        double theory = steady_state_mispredict_rate(p);
        double sim = simulate_mispredict_rate(p, N);
        printf("p=%.2f theory=%.6f sim=%.6f abs_diff=%.6f\n", p, theory, sim, std::fabs(theory - sim));
    }
    return 0;
}
