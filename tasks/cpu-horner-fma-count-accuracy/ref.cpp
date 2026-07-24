#include <cmath>
#include "sol.hpp"

void horner_eval(const double* coeffs, int n_coeffs, double x, double* value_out, long* fma_count_out) {
    int d = n_coeffs - 1;
    double result = coeffs[d];
    long count = 0;
    for (int i = d - 1; i >= 0; i--) {
        result = std::fma(result, x, coeffs[i]);
        count++;
    }
    *value_out = result;
    *fma_count_out = count;
}
