#include "sol.hpp"
#include <cstdio>

struct Scenario {
    long n; int num_ops; double flops_per_op; int elem_bytes;
    double peak_gflops; double peak_gbps;
};

// FIXED driver. 6 scenarios: fusion helps but not enough to flip regime,
// fusion flips memory-bound -> compute-bound (several different ridge
// points / op counts), fusion still not enough even with 5 fused ops,
// and one EDGE case where the fused AI lands EXACTLY on the ridge point
// (">=" must count as compute-bound, not ">").
int main() {
    static const Scenario scenarios[] = {
        {100000, 3, 2.0,  4, 200.0, 100.0},  // ai 0.25->0.75, ridge 2.0: no flip
        {100000, 3, 8.0,  4, 200.0, 100.0},  // ai 1.0->3.0,  ridge 2.0: FLIP
        {50000,  3, 20.0, 4, 200.0, 50.0},   // ai 2.5->7.5,  ridge 4.0: FLIP
        {200000, 5, 1.0,  8, 100.0, 200.0},  // ai 0.0625->0.3125, ridge 0.5: no flip
        {20000,  3, 50.0, 4, 100.0, 10.0},   // ai 6.25->18.75, ridge 10.0: FLIP
        {1000,   4, 8.0,  4, 100.0, 25.0},   // ai 1.0->4.0, ridge 4.0 EXACT: FLIP
    };

    for (const auto& s : scenarios) {
        FusionResult r = fusion_ai_and_flip(s.n, s.num_ops, s.flops_per_op,
                                             s.elem_bytes, s.peak_gflops, s.peak_gbps);
        printf("n=%ld ops=%d unfused_ai=%.6f fused_ai=%.6f "
               "unfused_cb=%d fused_cb=%d flipped=%d\n",
               s.n, s.num_ops, r.unfused_ai, r.fused_ai,
               (int)r.unfused_compute_bound, (int)r.fused_compute_bound,
               (int)r.regime_flipped);
    }
    return 0;
}
