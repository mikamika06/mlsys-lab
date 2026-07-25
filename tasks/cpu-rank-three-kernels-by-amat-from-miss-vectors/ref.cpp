#include "sol.hpp"

void rank_by_amat(const long* accesses, const long* l1_misses, const long* l2_misses,
                   double* amat_out, int* rank_out) {
    const double L1_HIT = 1.0, L2_HIT = 10.0, MEM_PENALTY = 100.0;
    for (int k = 0; k < 3; k++) {
        double l1mr = (double)l1_misses[k] / (double)accesses[k];
        double l2mr = (double)l2_misses[k] / (double)l1_misses[k];
        amat_out[k] = L1_HIT + l1mr * (L2_HIT + l2mr * MEM_PENALTY);
    }
    rank_out[0] = 0; rank_out[1] = 1; rank_out[2] = 2;
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (amat_out[rank_out[j]] < amat_out[rank_out[i]]) {
                int tmp = rank_out[i]; rank_out[i] = rank_out[j]; rank_out[j] = tmp;
            }
        }
    }
}
