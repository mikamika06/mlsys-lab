#pragma once

// Roofline ridge point: the arithmetic intensity (FLOP/byte) at which a
// device's peak-compute line crosses its peak-bandwidth line.
//
//   ridge = peak_flops / peak_bw
//
// peak_flops: device peak floating-point throughput, in FLOP/s.
// peak_bw:    device peak memory bandwidth, in bytes/s.
//
// A kernel whose own arithmetic intensity is BELOW this value is
// memory-bound on this device (moving data is the bottleneck); at or
// above it, the kernel is compute-bound (arithmetic is the bottleneck).
// Return the ridge point in FLOP/byte.
double ridge_point(double peak_flops, double peak_bw);
