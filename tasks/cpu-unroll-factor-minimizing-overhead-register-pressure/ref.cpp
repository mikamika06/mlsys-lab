#include "sol.hpp"

long unroll_cost(int N, int U, int C_loop, int R, int C_spill) {
    long outer_iters = (N + U - 1) / U;
    int over = U - R;
    long spill = (over > 0) ? (long)over * C_spill : 0;
    return outer_iters * (C_loop + spill);
}

int choose_best_unroll(int N, int max_U, int C_loop, int R, int C_spill) {
    int best_u = 1;
    long best_cost = unroll_cost(N, 1, C_loop, R, C_spill);
    for (int u = 2; u <= max_U; u++) {
        long c = unroll_cost(N, u, C_loop, R, C_spill);
        if (c < best_cost) {
            best_cost = c;
            best_u = u;
        }
    }
    return best_u;
}
