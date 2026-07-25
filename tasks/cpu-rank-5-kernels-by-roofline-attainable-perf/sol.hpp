#pragma once

// A kernel's arithmetic intensity is AI = flops / bytes (FLOP per byte
// moved). Its ATTAINABLE throughput under the roofline model is capped
// by whichever resource runs out first:
//
//   attainable = min(peak_flops_per_sec, AI * peak_bytes_per_sec)
//
// (peak_bytes_per_sec is the machine's peak memory bandwidth -- the same
// value for every kernel, since it's a property of the machine, not the
// kernel.)
//
// Rank `n` kernels by attainable throughput, HIGHEST first. Write the
// kernels' ORIGINAL indices (0-based, into `flops`/`bytes`), in ranked
// (descending attainable) order, into rank_out[0..n). No two kernels tie
// on attainable throughput in any graded fixture.
void rank_kernels_by_attainable_perf(const double* flops, const double* bytes, int n,
                                      double peak_flops_per_sec, double peak_bytes_per_sec,
                                      int* rank_out);
