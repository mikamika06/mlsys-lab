#include "sol.hpp"

// TODO: compute AMAT[k] = L1_HIT + L1_miss_rate*(L2_HIT +
// L2_miss_rate*MEM_PENALTY) for each of the 3 kernels, then write the
// kernel ids sorted by ascending AMAT into rank_out[0..3). See sol.hpp.
void rank_by_amat(const long* accesses, const long* l1_misses, const long* l2_misses,
                   double* amat_out, int* rank_out) {
    (void)accesses; (void)l1_misses; (void)l2_misses;
    for (int k = 0; k < 3; k++) { amat_out[k] = 0.0; rank_out[k] = k; }
    // your code here
}
