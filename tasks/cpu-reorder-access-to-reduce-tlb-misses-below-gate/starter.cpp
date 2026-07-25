#include <cstddef>
#include "sol.hpp"

// Straightforward column-major sweep: correct, and a completely reasonable
// first implementation if you just wrote out "for every column, for every
// row" -- but it revisits the SAME set of pages (all of them, once per
// column) far more often than the TLB can hold at once.
double sum_matrix_reordered(const double* data, int R, int C, int ld) {
    double sum = 0.0;
    for (int j = 0; j < C; ++j) {
        for (int i = 0; i < R; ++i) {
            const double* p = &data[static_cast<size_t>(i) * ld + j];
            touch_page(p);
            sum += *p;
        }
    }
    return sum;
}
