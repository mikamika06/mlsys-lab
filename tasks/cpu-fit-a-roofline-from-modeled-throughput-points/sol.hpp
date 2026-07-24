#pragma once

// A set of (arithmetic intensity, attained throughput) measurements was
// generated from the roofline model
//
//   attained = min(peak_flops, ai * peak_bw)
//
// The fixture guarantees at least one point is deep in the memory-bound
// region (small ai, where attained == ai * peak_bw exactly) and at least
// one point is deep in the compute-bound plateau (large ai, where
// attained == peak_flops exactly).
//
// fit_roofline: given n samples ai[0..n) / attained[0..n), estimate the
// two roofline parameters:
//   - find the sample with the SMALLEST ai; that sample is memory-bound,
//     so *peak_bw_out = attained[that] / ai[that] (the slope of the
//     memory-bound line).
//   - find the sample with the LARGEST ai; that sample is on the
//     compute-bound plateau, so *peak_flops_out = attained[that]
//     directly (the height of the plateau).
void fit_roofline(const double* ai, const double* attained, int n, double* peak_bw_out, double* peak_flops_out);
