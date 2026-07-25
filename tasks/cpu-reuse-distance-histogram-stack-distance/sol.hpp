#pragma once

// ============================================================================
// For each access i in addrs[0..n), let line_i = addrs[i] / line_bytes.
// The STACK DISTANCE of access i is the number of DISTINCT lines accessed
// strictly between the PREVIOUS access to line_i (exclusive) and access i
// (exclusive) -- how many different other lines were touched since
// line_i was last referenced, not counting line_i itself. If line_i has
// never been accessed before access i, it is a COLD access (no stack
// distance).
//
// The trace only ever touches `num_lines` distinct lines, so no stack
// distance can exceed num_lines - 1 (there simply aren't more OTHER
// distinct lines to have been touched in between).
//
// Fill hist_out[0..num_lines]:
//   hist_out[0]     = number of COLD accesses
//   hist_out[1 + d] = number of accesses with stack distance EXACTLY d,
//                     for d in [0, num_lines - 1]
// ============================================================================
void stack_distance_histogram(const long* addrs, int n, int line_bytes,
                               int num_lines, long* hist_out);
