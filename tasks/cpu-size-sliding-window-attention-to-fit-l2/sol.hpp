#pragma once

// ============================================================================
// LEARNER implements both of these in solve.cpp.
//
// Sliding-window attention for one query, with window size W, keeps the
// following resident at once (D = head dim, elem_bytes = bytes per K/V
// element, score_bytes = bytes per attention-score element):
//
//   - the query vector Q:            D * elem_bytes             (fixed)
//   - the output accumulator:        D * elem_bytes             (fixed)
//   - the W keys in the window:      W * D * elem_bytes
//   - the W values in the window:    W * D * elem_bytes
//   - the W attention scores:        W * score_bytes
//
// attention_working_set_bytes(W, D, elem_bytes, score_bytes):
//   Return the total: 2*D*elem_bytes + W*(2*D*elem_bytes + score_bytes).
//
// choose_max_window(l2_capacity_bytes, D, elem_bytes, score_bytes):
//   Return the LARGEST W >= 0 such that
//   attention_working_set_bytes(W, D, elem_bytes, score_bytes) <=
//   l2_capacity_bytes (i.e. the biggest window that still keeps the whole
//   working set L2-resident). Must call attention_working_set_bytes(...)
//   to do it (directly, or via an equivalent closed-form derived from the
//   same formula) -- not a separately hand-tuned computation.
// ============================================================================
long attention_working_set_bytes(int W, int D, int elem_bytes, int score_bytes);
int choose_max_window(long l2_capacity_bytes, int D, int elem_bytes, int score_bytes);
