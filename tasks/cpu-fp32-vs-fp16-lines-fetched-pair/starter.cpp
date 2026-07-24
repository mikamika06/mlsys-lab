#include "sol.hpp"

// TODO: for width in {4, 2}, count the distinct values of
// (base + i*width) / line_bytes over i in [0, n). See sol.hpp.
void compare_fp32_fp16_lines(long base, int n, int line_bytes, long* out) {
    (void)base; (void)n; (void)line_bytes;
    out[0] = 0;
    out[1] = 0;
    // your code here
}
