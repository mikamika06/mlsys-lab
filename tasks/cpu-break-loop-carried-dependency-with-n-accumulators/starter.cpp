#include "sol.hpp"

// TODO: use `num_accumulators` INDEPENDENT accumulators (e.g. accumulator
// j takes every num_accumulators-th element starting at index j), then
// combine the partial sums. Do not just sum everything into one running
// Tracked value -- that has the correct .value but a much longer .depth.
Tracked reduce_with_accumulators(const Tracked* x, int n, int num_accumulators) {
    (void)num_accumulators;
    Tracked total{0.0, 0};
    for (int i = 0; i < n; i++) {
        total = total + x[i];
    }
    return total;
}
