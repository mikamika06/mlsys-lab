#include "sol.hpp"

// TODO: implement both SAXPY variants using load_f/store_f for every
// access. See sol.hpp for the exact per-iteration op-count contract.
void saxpy_unhoisted(const float* a_ptr, const float* x, float* y, int n) {
    (void)a_ptr;
    (void)x;
    (void)y;
    (void)n;
}

void saxpy_hoisted(const float* a_ptr, const float* x, float* y, int n) {
    (void)a_ptr;
    (void)x;
    (void)y;
    (void)n;
}
