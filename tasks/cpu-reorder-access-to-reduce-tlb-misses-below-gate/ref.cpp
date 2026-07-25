#include <cstddef>
#include "sol.hpp"

// Row-blocked traversal: split the R rows into blocks of kRowBlock rows.
// For each block, sweep every column for every row IN THAT BLOCK before
// moving to the next block. A block's rows only span a handful of pages
// (far fewer than a full column sweep of the whole matrix would), so once
// those pages are loaded on the block's first column, every later column
// in the same block finds them still resident in the TLB.
double sum_matrix_reordered(const double* data, int R, int C, int ld) {
    constexpr int kRowBlock = 128;

    double sum = 0.0;
    for (int i0 = 0; i0 < R; i0 += kRowBlock) {
        int i1 = (i0 + kRowBlock < R) ? i0 + kRowBlock : R;
        for (int j = 0; j < C; ++j) {
            for (int i = i0; i < i1; ++i) {
                const double* p = &data[static_cast<size_t>(i) * ld + j];
                touch_page(p);
                sum += *p;
            }
        }
    }
    return sum;
}
