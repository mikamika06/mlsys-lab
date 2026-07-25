#include "sol.hpp"

// TODO: base + (i*row_stride + j*col_stride) * elem_bytes -- see sol.hpp.
long element_addr(long base, int i, int j, long row_stride, long col_stride, int elem_bytes) {
    (void)base; (void)i; (void)j; (void)row_stride; (void)col_stride; (void)elem_bytes;
    // your code here
    return 0;
}

// TODO: walk the traversal in the given order, counting a "fetch" every
// time the touched line differs from the immediately preceding access
// -- see sol.hpp.
long traversal_fetch_count(long base, int R, int C, long row_stride, long col_stride,
                            int elem_bytes, int line_bytes, bool row_major) {
    (void)base; (void)R; (void)C; (void)row_stride; (void)col_stride;
    (void)elem_bytes; (void)line_bytes; (void)row_major;
    // your code here
    return 0;
}
