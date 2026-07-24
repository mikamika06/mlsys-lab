#pragma once

// Arithmetic intensity (AI) = total FLOPs / total bytes moved to/from
// memory. The roofline model's "ridge point" is peak_gflops / peak_gbps
// (FLOP/byte); a kernel is compute-bound if its AI is at or above the
// ridge point, memory-bound if below.
//
// A chain of `num_ops` elementwise ops, each doing `flops_per_op` FLOPs
// per element, is applied to an array of `n` elements (`elem_bytes` bytes
// each):
//
//   UNFUSED (each op is a separate pass over memory): op i reads its
//   input array and writes its own output array, so every op contributes
//   one full read + one full write of n elements:
//     total_flops    = n * num_ops * flops_per_op
//     unfused_bytes  = 2 * n * num_ops * elem_bytes
//     unfused_ai     = total_flops / unfused_bytes
//
//   FUSED (all num_ops chained inside ONE pass, each output staying in a
//   register instead of going back out to memory): only the very first
//   input is read from memory and only the very last output is written --
//   num_ops does not change how many bytes move, only how many FLOPs are
//   spent per byte:
//     fused_bytes    = 2 * n * elem_bytes
//     fused_ai       = total_flops / fused_bytes
//
// Both AIs are compared against the SAME ridge point (the compute/memory
// balance point of the machine does not change just because the kernel
// was rewritten).
struct FusionResult {
    double unfused_ai;
    double fused_ai;
    bool unfused_compute_bound;  // unfused_ai >= ridge_point
    bool fused_compute_bound;    // fused_ai >= ridge_point
    bool regime_flipped;         // unfused_compute_bound != fused_compute_bound
};

FusionResult fusion_ai_and_flip(long n, int num_ops, double flops_per_op,
                                 int elem_bytes, double peak_gflops, double peak_gbps);
