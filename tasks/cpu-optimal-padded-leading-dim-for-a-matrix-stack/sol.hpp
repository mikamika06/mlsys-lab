#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// A stack of matrices is laid out back-to-back in memory, each one
// `M` rows tall, row-major, with every row padded out to `ld` elements
// (`ld >= n_cols`, the matrix's true column count) of `elem_bytes` bytes
// each. So matrix `k`'s row `r`, column 0 lives at byte address
// `k * (M * ld * elem_bytes) + r * (ld * elem_bytes)`, i.e. every matrix
// in the stack occupies exactly `M * ld * elem_bytes` bytes.
//
// A deterministic set-associative cache maps a byte address to a set via
// `(address / line_bytes) % num_sets`. If the per-matrix stride
// `M * ld * elem_bytes` happens to be an exact multiple of
// `line_bytes * num_sets`, then EVERY matrix's row-`r` element lands in
// the exact SAME set, no matter which matrix -- a periodic conflict alias
// that thrashes the moment the stack is deeper than the cache's
// associativity, regardless of how much total cache capacity is free.
//
// Return the SMALLEST `ld >= n_cols` for which
// `(M * ld * elem_bytes) % (line_bytes * num_sets) != 0` -- i.e. the
// smallest padding that breaks that exact periodic alias.
// ============================================================================
int choose_padded_ld(int n_cols, int M, int elem_bytes, int line_bytes, int num_sets);
