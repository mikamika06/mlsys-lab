#include "sol.hpp"

// TODO: total_flops = n*num_ops*flops_per_op; unfused_bytes =
// 2*n*num_ops*elem_bytes; fused_bytes = 2*n*elem_bytes; each ai =
// total_flops / its bytes; compute_bound = ai >= (peak_gflops/peak_gbps);
// flipped = unfused_compute_bound != fused_compute_bound. See sol.hpp.
FusionResult fusion_ai_and_flip(long n, int num_ops, double flops_per_op,
                                 int elem_bytes, double peak_gflops, double peak_gbps) {
    (void)n; (void)num_ops; (void)flops_per_op;
    (void)elem_bytes; (void)peak_gflops; (void)peak_gbps;
    // your code here
    FusionResult r;
    r.unfused_ai = 0.0;
    r.fused_ai = 0.0;
    r.unfused_compute_bound = false;
    r.fused_compute_bound = false;
    r.regime_flipped = false;
    return r;
}
