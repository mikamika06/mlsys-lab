#include "sol.hpp"

double sum_matrix_tlb_friendly(const double* values, long base, int R, int C) {
    double sum = 0.0;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            long addr = base + (long)(r * C + c) * 8;
            touch_page(addr);
            sum += values[r * C + c];
        }
    }
    return sum;
}
