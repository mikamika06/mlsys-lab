#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// The roofline model caps a kernel's attainable throughput by BOTH the
// machine's peak compute rate and its peak memory bandwidth times the
// kernel's own arithmetic intensity (FLOPs performed per byte moved from
// DRAM) — whichever is lower:
//
//     attainable = min(peak_flops, arithmetic_intensity * peak_bandwidth)
//
// peak_flops: the machine's peak compute throughput (FLOP/s).
// peak_bandwidth: the machine's peak DRAM bandwidth (bytes/s).
// arithmetic_intensity: FLOPs performed per byte moved (FLOPs/byte).
// ============================================================================
double attainable_flops(double peak_flops, double peak_bandwidth, double arithmetic_intensity);
