#include "sol.hpp"

// Reference: num_accumulators independent (strided) accumulators, each its
// own addition chain, combined at the end.
Tracked reduce_with_accumulators(const Tracked* x, int n, int num_accumulators) {
    int k = num_accumulators;
    Tracked* partial = new Tracked[k];
    for (int j = 0; j < k; j++) partial[j] = Tracked{0.0, 0};

    for (int i = 0; i < n; i++) {
        partial[i % k] = partial[i % k] + x[i];
    }

    Tracked total = partial[0];
    for (int j = 1; j < k; j++) total = total + partial[j];

    delete[] partial;
    return total;
}
