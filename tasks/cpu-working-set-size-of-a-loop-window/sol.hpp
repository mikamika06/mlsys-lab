#pragma once

// Denning's working-set model: the working set of a window of W
// consecutive memory accesses is the set of distinct cache lines that
// window touches. Its size in BYTES (distinct_lines * line_bytes) is
// the minimum cache capacity that would let those W accesses run
// without a single capacity miss.
//
// Given a trace addrs[0..n) of byte addresses, a window width W
// (1 <= W <= n), and line_bytes, return the LARGEST working-set size in
// bytes over EVERY contiguous window of W consecutive accesses in the
// trace -- i.e.
//   max over t in [0, n-W] of:
//     line_bytes * (number of distinct values of addrs[i] / line_bytes,
//                    for i in [t, t+W))
//
// This is the peak cache capacity the loop this trace came from ever
// needs at once -- not the same as counting distinct lines across the
// WHOLE trace (that ignores that old lines can be evicted once they
// fall out of the window) and not the same as checking only the first
// window (an early quiet window can undersell a later, more scattered
// one).
long max_working_set_bytes(const long* addrs, int n, int line_bytes, int W);
