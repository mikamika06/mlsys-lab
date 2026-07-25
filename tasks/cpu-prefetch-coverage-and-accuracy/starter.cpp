#include "sol.hpp"

// TODO: coverage = useful_prefetches / baseline_misses;
//       accuracy = useful_prefetches / total_prefetches. See sol.hpp.
void compute_coverage_accuracy(long baseline_misses, long total_prefetches, long useful_prefetches,
                                double* coverage_out, double* accuracy_out) {
    (void)baseline_misses; (void)total_prefetches; (void)useful_prefetches;
    *coverage_out = 0.0;
    *accuracy_out = 0.0;
    // your code here
}
