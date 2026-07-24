#include "sol.hpp"

// TODO: split the reduction across `num_chains` independent accumulators
// (round-robin: element i -> chain i % num_chains), calling
// report_op(i % num_chains) once per element, then sum the chains'
// partials together at the end. See sol.hpp.
double dot_product_ilp(const double* a, const double* b, int n, int num_chains) {
    (void)a; (void)b; (void)n; (void)num_chains;
    // your code here
    return 0.0;
}
