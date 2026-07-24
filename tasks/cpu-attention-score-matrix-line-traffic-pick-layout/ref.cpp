#include "sol.hpp"

void simulate_score_matrix_traffic(int layout, int seq_len, int head_dim,
                                    long q_base, long k_base) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            for (int d = 0; d < head_dim; d++) {
                long q_addr = q_base + (long)(i * head_dim + d) * 4;
                touch_byte(q_addr);

                long k_addr;
                if (layout == 0) {
                    k_addr = k_base + (long)(j * head_dim + d) * 4;
                } else {
                    k_addr = k_base + (long)(d * seq_len + j) * 4;
                }
                touch_byte(k_addr);
            }
        }
    }
}

int pick_better_layout(int seq_len, int head_dim) {
    const long Q_BASE = 0;
    const long K_BASE = 1000000;

    reset_cache();
    simulate_score_matrix_traffic(0, seq_len, head_dim, Q_BASE, K_BASE);
    long misses0 = miss_count();

    reset_cache();
    simulate_score_matrix_traffic(1, seq_len, head_dim, Q_BASE, K_BASE);
    long misses1 = miss_count();

    return (misses1 < misses0) ? 1 : 0;
}
