#pragma once

// A working-set access TRACE `addrs[0..n)` (byte addresses) is scanned
// with windows of every size w in [1, max_w]: window w is the LAST w
// elements of the trace, addrs[n-w .. n-1]. For each w, count the number
// of DISTINCT CACHE LINES (addr / line_bytes) among those w addresses
// and write it into curve_out[w-1].
//
// The curve is non-decreasing in w: a longer window can only include
// MORE addresses, so it can never see FEWER distinct lines than a
// shorter one. Once the window is at least as large as the trace's true
// working-set size K, every additional address it picks up is a REPEAT
// of a line already seen, so the curve stops growing there and stays
// flat (a "plateau") no matter how much larger w gets.
//
// Return the PLATEAU INDEX: the smallest w (1-indexed) such that
// curve_out[w-1] equals the curve's final value curve_out[max_w-1] --
// the point past which growing the window further never reveals another
// distinct line. For a trace that's exactly K lines repeating in a fixed
// cycle, this recovers K without ever being told K directly.
int plateau_index(const long* addrs, int n, int max_w, int line_bytes, int* curve_out);
