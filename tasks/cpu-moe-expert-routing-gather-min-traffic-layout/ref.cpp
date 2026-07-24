#include "sol.hpp"

void moe_gather(const double* weights, const int* expert_id, int T, int W, int E, long base, double* out) {
    // Grouped by expert: every token routed to expert 0, then expert 1, ...
    for (int e = 0; e < E; e++) {
        for (int t = 0; t < T; t++) {
            if (expert_id[t] != e) continue;
            double sum = 0.0;
            for (int w = 0; w < W; w++) {
                long addr = base + (long)(e * W + w) * 4;
                touch_byte(addr);
                sum += weights[e * W + w];
            }
            out[t] = sum;
        }
    }
}
