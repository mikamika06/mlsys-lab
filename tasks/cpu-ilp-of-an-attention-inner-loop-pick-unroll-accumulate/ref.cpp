#include "sol.hpp"

double dot_product_ilp(const double* a, const double* b, int n, int num_chains) {
    double acc[64] = {0};
    for (int i = 0; i < n; i++) {
        int chain = i % num_chains;
        acc[chain] += a[i] * b[i];
        report_op(chain);
    }
    double total = 0.0;
    for (int c = 0; c < num_chains; c++) total += acc[c];
    return total;
}
