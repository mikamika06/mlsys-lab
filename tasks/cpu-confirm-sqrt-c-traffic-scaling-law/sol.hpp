#pragma once

// Fit the exponent b of a power law y = a * x^b from n positive samples
// (x[i], y[i]), by ordinary least squares on the log-log transformed data:
//
//   X_i = ln(x[i]),  Y_i = ln(y[i])
//   b = sum_i (X_i - mean(X)) * (Y_i - mean(Y))  /  sum_i (X_i - mean(X))^2
//
// (the standard "fit a line, the slope is the exponent" trick: taking logs
// turns y = a*x^b into a straight line ln(y) = ln(a) + b*ln(x)).
//
// Return b.
double fit_scaling_exponent(const double* x, const double* y, int n);
