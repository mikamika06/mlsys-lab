#include "sol.hpp"

long element_addr(long base, int i, int j, long row_stride, long col_stride, int elem_bytes) {
    return base + (long)(i * row_stride + j * col_stride) * elem_bytes;
}

long traversal_fetch_count(long base, int R, int C, long row_stride, long col_stride,
                            int elem_bytes, int line_bytes, bool row_major) {
    long fetches = 0;
    bool have_open = false;
    long open_line = -1;

    auto visit = [&](int i, int j) {
        long addr = element_addr(base, i, j, row_stride, col_stride, elem_bytes);
        long line = addr / line_bytes;
        if (!have_open || line != open_line) {
            fetches++;
            open_line = line;
            have_open = true;
        }
    };

    if (row_major) {
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++) visit(i, j);
    } else {
        for (int j = 0; j < C; j++)
            for (int i = 0; i < R; i++) visit(i, j);
    }
    return fetches;
}
