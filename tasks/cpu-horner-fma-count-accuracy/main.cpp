#include <cstdio>
#include "sol.hpp"

// FIXED driver. Degree-5 polynomial (6 coefficients), evaluated at a
// fixed x.
int main() {
    const int n_coeffs = 6;
    double coeffs[n_coeffs] = {2.0, -3.0, 0.5, 1.25, -0.75, 4.0};
    const double x = 1.37;

    double value = -999.0;   // sentinel: an empty starter leaves this untouched
    long fma_count = -1;

    horner_eval(coeffs, n_coeffs, x, &value, &fma_count);

    printf("value=%.10f fma_count=%ld\n", value, fma_count);
    return 0;
}
