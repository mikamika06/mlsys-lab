#include <cmath>
#include "sol.hpp"

double fit_scaling_exponent(const double* x, const double* y, int n) {
    double mean_x = 0.0, mean_y = 0.0;
    double lx[64], ly[64];  // n is always small in this task
    for (int i = 0; i < n; i++) {
        lx[i] = std::log(x[i]);
        ly[i] = std::log(y[i]);
        mean_x += lx[i];
        mean_y += ly[i];
    }
    mean_x /= n;
    mean_y /= n;

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        num += (lx[i] - mean_x) * (ly[i] - mean_y);
        den += (lx[i] - mean_x) * (lx[i] - mean_x);
    }
    return num / den;
}
