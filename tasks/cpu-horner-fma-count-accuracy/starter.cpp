#include "sol.hpp"

// TODO: Horner's method with fma(). result = coeffs[d]; for i = d-1
// downto 0: result = std::fma(result, x, coeffs[i]), counting each call.
// See sol.hpp.
void horner_eval(const double* coeffs, int n_coeffs, double x, double* value_out, long* fma_count_out) {
    (void)coeffs; (void)n_coeffs; (void)x;
    *value_out = 0.0;
    *fma_count_out = 0;
    // your code here
}
