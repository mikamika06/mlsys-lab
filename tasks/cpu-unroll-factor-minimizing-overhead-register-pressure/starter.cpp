#include "sol.hpp"

// TODO: unroll_cost = ceil(N/U) * (C_loop + max(0,U-R)*C_spill).
// choose_best_unroll: try U=1..max_U, return the U with smallest
// unroll_cost (smallest U on a tie).
long unroll_cost(int N, int U, int C_loop, int R, int C_spill) {
    (void)N; (void)U; (void)C_loop; (void)R; (void)C_spill;
    // your code here
    return 0;
}

int choose_best_unroll(int N, int max_U, int C_loop, int R, int C_spill) {
    (void)N; (void)max_U; (void)C_loop; (void)R; (void)C_spill;
    // your code here
    return 1;
}
