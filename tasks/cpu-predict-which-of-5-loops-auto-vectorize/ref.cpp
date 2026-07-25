#include "sol.hpp"

// Elementwise ops with no loop-carried dependency vectorize; a genuine
// carried recurrence, unsafe float reassociation in a plain sum, and a
// non-uniform (quadratic) index all block it; a branch-free compare+select
// vectorizes fine.
bool predictLoop1() { return true; }   // elementwise_add
bool predictLoop2() { return false; }  // carried_dep
bool predictLoop3() { return false; }  // plain_sum
bool predictLoop4() { return true; }   // branch_free_select
bool predictLoop5() { return false; }  // nonuniform_index
