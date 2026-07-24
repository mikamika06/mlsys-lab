#include "sol.hpp"

void accumulate_aliased(double* dest, const double* src, int n) {
    for (int i = 0; i < n; ++i) {
        double d = load_double(dest);
        double s = load_double(&src[i]);
        store_double(dest, d + s);
    }
}

void accumulate_hoisted(double* dest, const double* src, int n) {
    double acc = load_double(dest);
    for (int i = 0; i < n; ++i) {
        acc += load_double(&src[i]);
    }
    store_double(dest, acc);
}
