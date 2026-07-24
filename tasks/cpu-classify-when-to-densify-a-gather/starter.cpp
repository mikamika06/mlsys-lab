#include "sol.hpp"

// TODO: simulate GATHER (k direct touches of ORIG_BASE + indices[i]*elem_bytes)
// and DENSIFY (touch each distinct index once ascending at ORIG_BASE, then
// touch SCRATCH_BASE + rank*elem_bytes for each of the k requests) against
// the shared cache_reset()/touch() model, and return whichever strategy
// measured fewer total misses (GATHER on a tie).
int classify_gather_strategy(const long* indices, int k, long elem_bytes) {
    (void)indices; (void)k; (void)elem_bytes;
    // your code here
    return GATHER;
}
