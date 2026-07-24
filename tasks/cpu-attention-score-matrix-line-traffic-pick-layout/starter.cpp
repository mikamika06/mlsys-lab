#include "sol.hpp"

// TODO: implement simulate_score_matrix_traffic and pick_better_layout per
// sol.hpp. Right now neither touches any memory, so both reported miss
// counts stay 0 and the layout choice is meaningless.
void simulate_score_matrix_traffic(int layout, int seq_len, int head_dim,
                                    long q_base, long k_base) {
    (void)layout; (void)seq_len; (void)head_dim; (void)q_base; (void)k_base;
    // your code here
}

int pick_better_layout(int seq_len, int head_dim) {
    (void)seq_len; (void)head_dim;
    return 0;  // your code here
}
