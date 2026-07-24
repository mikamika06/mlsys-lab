#include "sol.hpp"

// TODO: process tokens GROUPED BY EXPERT (all tokens of expert 0, then
// all of expert 1, ...), touching that expert's W simulated addresses
// and summing its real weights, writing out[t] for each token. See
// sol.hpp.
void moe_gather(const double* weights, const int* expert_id, int T, int W, int E, long base, double* out) {
    (void)weights; (void)expert_id; (void)T; (void)W; (void)E; (void)base; (void)out;
    // your code here
}
