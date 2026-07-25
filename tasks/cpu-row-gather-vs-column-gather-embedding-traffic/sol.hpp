#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// An embedding table has `V` rows (vocabulary entries) and `D` columns
// (embedding dimensions), `elem_bytes` bytes per element. Gathering
// `idx[0..k)` means fetching the FULL D-dimensional vector for each of
// those `k` row indices. Compute the total memory TRAFFIC of that gather
// -- the number of DISTINCT `line_bytes`-byte cache lines touched, across
// every element of every gathered vector combined -- under two different
// physical layouts of the SAME table:
//
//   out[0] ROW-MAJOR:    element (v, d) lives at byte offset
//                        v*D*elem_bytes + d*elem_bytes.
//                        (each vector is one contiguous D*elem_bytes run)
//
//   out[1] COLUMN-MAJOR: element (v, d) lives at byte offset
//                        d*V*elem_bytes + v*elem_bytes.
//                        (each DIMENSION is one contiguous V*elem_bytes
//                        run; a single vector's D elements are scattered
//                        one per dimension-block, V*elem_bytes apart)
//
// For each layout: touch every (v, d) pair for v in idx[0..k) and d in
// [0, D), compute its byte address under that layout, and count the
// number of DISTINCT values of (address / line_bytes) across the whole
// gather (a line touched by two different elements only counts once).
// Write the row-major count to out[0] and the column-major count to
// out[1].
// ============================================================================
void gather_line_traffic(const int* idx, int k, int V, int D, int elem_bytes, int line_bytes, long* out);
