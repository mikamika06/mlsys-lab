#include <set>
#include "sol.hpp"

void gather_line_traffic(const int* idx, int k, int V, int D, int elem_bytes, int line_bytes, long* out) {
    std::set<long> row_lines, col_lines;
    for (int i = 0; i < k; i++) {
        int v = idx[i];
        for (int d = 0; d < D; d++) {
            long row_addr = (long)v * D * elem_bytes + (long)d * elem_bytes;
            row_lines.insert(row_addr / line_bytes);

            long col_addr = (long)d * V * elem_bytes + (long)v * elem_bytes;
            col_lines.insert(col_addr / line_bytes);
        }
    }
    out[0] = (long)row_lines.size();
    out[1] = (long)col_lines.size();
}
