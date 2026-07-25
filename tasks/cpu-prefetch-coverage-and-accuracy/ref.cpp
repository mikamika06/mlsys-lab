#include "sol.hpp"

void compute_coverage_accuracy(long baseline_misses, long total_prefetches, long useful_prefetches,
                                double* coverage_out, double* accuracy_out) {
    *coverage_out = (double)useful_prefetches / (double)baseline_misses;
    *accuracy_out = (double)useful_prefetches / (double)total_prefetches;
}
