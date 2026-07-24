#include "sol.hpp"

FusionResult fusion_ai_and_flip(long n, int num_ops, double flops_per_op,
                                 int elem_bytes, double peak_gflops, double peak_gbps) {
    double total_flops = (double)n * (double)num_ops * flops_per_op;
    double unfused_bytes = 2.0 * (double)n * (double)num_ops * (double)elem_bytes;
    double fused_bytes = 2.0 * (double)n * (double)elem_bytes;

    FusionResult r;
    r.unfused_ai = total_flops / unfused_bytes;
    r.fused_ai = total_flops / fused_bytes;

    double ridge = peak_gflops / peak_gbps;
    r.unfused_compute_bound = r.unfused_ai >= ridge;
    r.fused_compute_bound = r.fused_ai >= ridge;
    r.regime_flipped = r.unfused_compute_bound != r.fused_compute_bound;
    return r;
}
