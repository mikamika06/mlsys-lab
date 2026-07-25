#include "sol.hpp"

// TODO: model distinct 64-byte lines touched under AoS and under SoA for
// this access pattern, return 1 if soa_lines <= aos_lines else 0 -- see
// sol.hpp.
int soa_is_optimal(int N, int F, const int* field_bytes, const bool* mask) {
    (void)N; (void)F; (void)field_bytes; (void)mask;
    // your code here
    return 0;
}
