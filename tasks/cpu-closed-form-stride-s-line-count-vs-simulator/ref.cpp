#include "sol.hpp"

long distinct_lines_stride_walk(long n, long stride, long elem_bytes, long line_bytes) {
    if (n <= 0) return 0;
    long L = line_bytes / elem_bytes;           // elements per line
    long last_line = ((n - 1) * stride) / L;     // line index of the final access
    long contiguous = last_line + 1;             // lines are hit with no gaps when stride <= L
    return contiguous < n ? contiguous : n;      // but never more distinct lines than accesses
}
