#pragma once

// A counted `for (i = 0; i < N; i++)` loop pays 2 overhead operations on
// EVERY iteration that are not part of the useful body:
//   - counter increment:      i += 1
//   - loop-condition branch:  i < N ?
// so K = 2 overhead ops/iteration, and the total overhead across the
// whole loop is 2 * N.
//
// Unrolling the loop by a factor U packs U copies of the body into each
// iteration, shrinking the iteration count from N down to N / U (integer
// division -- any leftover elements still need a remainder loop, but
// that remainder loop's own overhead is not part of this model). The
// data-element accesses stay exactly N either way; only the per-iteration
// bookkeeping shrinks.
//
// Return the number of overhead operations ELIMINATED by unrolling:
//   saved = 2 * (N - N / U)
long long unroll_overhead_saved(long long N, long long U);
