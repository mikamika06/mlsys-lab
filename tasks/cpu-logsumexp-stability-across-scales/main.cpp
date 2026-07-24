#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Every fixture is hand-built (no rand()/time()), spanning
// normal, huge-positive, huge-negative, and mixed-scale inputs, plus two
// small edge cases (single element, all-identical elements).

namespace {

void run(const char* name, const std::vector<double>& x) {
    double r = log_sum_exp(x);
    printf("%s = %.10f\n", name, r);
}

}  // namespace

int main() {
    run("normal", {1.0, 2.0, 3.0});
    run("large_positive", {1000.0, 1000.5, 999.0});
    run("large_negative", {-1000.0, -1000.5, -999.0});
    run("mixed_range", {-1000.0, 500.0, 1000.0});
    run("single", {42.0});
    run("identical", {5.0, 5.0, 5.0, 5.0, 5.0});
    return 0;
}
