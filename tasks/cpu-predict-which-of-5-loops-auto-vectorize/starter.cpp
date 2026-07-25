#include "sol.hpp"

// TODO: predict, for each of the 5 loops documented in sol.hpp, whether it
// autovectorizes at -O2 on this compiler.
bool predictLoop1() { return true; }   // elementwise_add
bool predictLoop2() { return false; }  // carried_dep
bool predictLoop3() { return true; }   // plain_sum -- WRONG: unsafe fp reassociation blocks it
bool predictLoop4() { return true; }   // branch_free_select
bool predictLoop5() { return true; }   // nonuniform_index -- WRONG: non-uniform stride blocks it
