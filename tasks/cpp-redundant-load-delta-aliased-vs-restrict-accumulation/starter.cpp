#include "sol.hpp"

// TODO: implement both loops using load_double/store_double for every
// access. See sol.hpp for the exact per-iteration op-count contract.
void accumulate_aliased(double* dest, const double* src, int n) {
    (void)dest;
    (void)src;
    (void)n;
}

void accumulate_hoisted(double* dest, const double* src, int n) {
    (void)dest;
    (void)src;
    (void)n;
}
