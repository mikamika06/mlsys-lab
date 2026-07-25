#include "sol.hpp"

// Row-major traversal (r outer, c inner) matches row-major storage:
// consecutive touches walk along a row, staying in the same or next
// line.
double sum_matrix(const double* values, long base, int R, int C) {
    double sum = 0.0;
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            long addr = base + (long)(r * C + c) * 4;
            touch_byte(addr);
            sum += values[r * C + c];
        }
    }
    return sum;
}
