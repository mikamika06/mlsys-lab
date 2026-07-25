#include <cstdio>
#include "sol.hpp"

// FIXED driver, two scenarios: one where the optimum lands exactly at the
// register count, one (prime N, so no exact-division ties) where a
// slightly larger unroll factor is still worth the spill cost.
namespace {
void run_scenario(int N, int max_U, int C_loop, int R, int C_spill) {
    int u = choose_best_unroll(N, max_U, C_loop, R, C_spill);
    long cost = unroll_cost(N, u, C_loop, R, C_spill);
    printf("N=%d C_loop=%d R=%d C_spill=%d -> U=%d cost=%ld\n", N, C_loop, R, C_spill, u, cost);
}
}  // namespace

int main() {
    run_scenario(1024, 64, 20, 8, 15);
    run_scenario(997, 64, 60, 6, 10);
    return 0;
}
