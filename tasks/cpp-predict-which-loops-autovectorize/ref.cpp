#include "sol.hpp"

// Elementwise ops vectorize (even without __restrict__, via a runtime alias
// check); anything with a genuine loop-carried dependency, unsafe float
// reassociation, a data-dependent exit, or an opaque call does not; this
// particular max-reduction idiom happens not to at -O2 either.
bool predictLoop1() { return true; }   // vectorizable_add
bool predictLoop2() { return true; }   // restrict_mul
bool predictLoop3() { return true; }   // int_add
bool predictLoop4() { return false; }  // reduction_dep
bool predictLoop5() { return false; }  // plain_sum_reduction
bool predictLoop6() { return false; }  // early_exit_loop
bool predictLoop7() { return false; }  // calls_opaque_fn
bool predictLoop8() { return false; }  // max_reduction
