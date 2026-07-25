#include "sol.hpp"

void stack_distance_histogram(const long* addrs, int n, int line_bytes,
                               int num_lines, long* hist_out) {
    // your code here
    (void)addrs;
    (void)n;
    (void)line_bytes;
    for (int i = 0; i <= num_lines; ++i) hist_out[i] = 0;
}
