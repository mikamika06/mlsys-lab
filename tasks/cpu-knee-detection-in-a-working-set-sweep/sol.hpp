#pragma once

// A "latency ladder" sweep measures average per-access latency while
// growing the working-set size from small to large: as long as the
// working set fits in one cache level, latency stays roughly flat (a
// PLATEAU, plus a little measurement jitter); once it outgrows that
// level, latency jumps up to the next level's cost (a KNEE) before
// plateauing again -- one knee per cache-hierarchy boundary crossed.
//
// Detect every knee in a latency array of `n` samples (indexed by
// increasing working-set size): index i (1 <= i < n) is a knee if the
// RELATIVE jump from sample i-1 to sample i exceeds rel_threshold:
//
//   (latency[i] - latency[i-1]) / latency[i-1] > rel_threshold
//
// Ordinary measurement jitter WITHIN a plateau produces small relative
// changes (well under rel_threshold) and must NOT be reported as a knee.
//
// Write the knee indices, in increasing order, into out_indices (a
// caller-provided buffer with room for at least n-1 ints) and return how
// many were found.
int detect_knees(const double* latency, int n, double rel_threshold, int* out_indices);
