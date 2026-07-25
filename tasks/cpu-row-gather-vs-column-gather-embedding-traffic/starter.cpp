#include "sol.hpp"

// TODO: for each layout, compute (v,d) -> byte address per sol.hpp, and
// count DISTINCT (address / line_bytes) values across the whole gather.
// out[0] = row-major distinct line count, out[1] = column-major.
void gather_line_traffic(const int* idx, int k, int V, int D, int elem_bytes, int line_bytes, long* out) {
    (void)idx; (void)k; (void)V; (void)D; (void)elem_bytes; (void)line_bytes;
    // your code here
}
