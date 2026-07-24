#pragma once

// Evaluate a degree-d polynomial with coefficients coeffs[0..n_coeffs)
// (coeffs[n_coeffs-1] is the highest-degree coefficient, coeffs[0] the
// constant term; d = n_coeffs - 1) at x, using Horner's method:
//
//   result = coeffs[d]
//   for i = d-1 downto 0: result = result * x + coeffs[i]
//
// Each step's multiply-then-add must be done as ONE fused multiply-add
// (std::fma(result, x, coeffs[i]), from <cmath>) rather than a separate
// multiply and add -- this both rounds once instead of twice (more
// accurate) and is a single hardware instruction (faster). Horner's
// method needs exactly d = n_coeffs - 1 such fma() calls, no more, no
// fewer.
//
// Write the evaluated value into *value_out and the number of fma()
// calls actually made into *fma_count_out.
void horner_eval(const double* coeffs, int n_coeffs, double x, double* value_out, long* fma_count_out);
