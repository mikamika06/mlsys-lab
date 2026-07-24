#include <algorithm>
#include <cmath>
#include "sol.hpp"

double log_sum_exp(const std::vector<double>& x) {
    double m = x[0];
    for (double v : x) m = std::max(m, v);

    double sum = 0.0;
    for (double v : x) sum += std::exp(v - m);

    return m + std::log(sum);
}
