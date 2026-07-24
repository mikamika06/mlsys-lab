#include "sol.hpp"

// TODO: predict, for each of the 8 loops documented in sol.hpp, whether it
// autovectorizes at -O2 on this compiler.
bool predictLoop1() { return true; }   // vectorizable_add
bool predictLoop2() { return true; }   // restrict_mul
bool predictLoop3() { return true; }   // int_add
bool predictLoop4() { return false; }  // reduction_dep
bool predictLoop5() { return true; }   // plain_sum_reduction -- WRONG: unsafe fp reassociation
bool predictLoop6() { return false; }  // early_exit_loop
bool predictLoop7() { return true; }   // calls_opaque_fn -- WRONG: opaque call blocks it
bool predictLoop8() { return true; }   // max_reduction -- WRONG: doesn't vectorize at -O2
