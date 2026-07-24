#pragma once

// Roofline model. A device has peak compute `peak_flops` (FLOP/s) and peak
// memory bandwidth `peak_bw` (bytes/s). For a kernel with arithmetic
// intensity `ai` (FLOP/byte), the roofline model bounds its attainable
// throughput by
//
//   attainable = min(peak_flops, ai * peak_bw)
//
// The RIDGE POINT is the intensity where the two terms are equal:
//   ridge = peak_flops / peak_bw   (FLOP/byte)
//
// Below the ridge point the memory term is the binding constraint (the
// kernel is MEMORY-BOUND: adding more FLOPs is free until ai reaches the
// ridge). At or above the ridge point the compute term binds (the kernel
// is COMPUTE-BOUND: more bandwidth would not help).
//
// classify_regimes: given peak_flops, peak_bw, and an array `ai` of n
// arithmetic-intensity values (FLOP/byte), write into out[i]:
//   1  if ai[i] >= ridge point   (compute-bound)
//   0  if ai[i] <  ridge point   (memory-bound)
void classify_regimes(double peak_flops, double peak_bw, const double* ai, int n, int* out);
