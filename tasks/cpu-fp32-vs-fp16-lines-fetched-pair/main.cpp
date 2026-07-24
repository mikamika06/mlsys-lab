#include <cstdio>
#include "sol.hpp"

// FIXED driver. 100 elements, 64-byte lines, base aligned to a line
// boundary.
int main() {
    const long base = 0;
    const int n = 100;
    const int line_bytes = 64;

    long out[2] = {-1, -1};  // sentinel: an empty starter leaves this untouched
    compare_fp32_fp16_lines(base, n, line_bytes, out);

    printf("fp32_lines=%ld fp16_lines=%ld\n", out[0], out[1]);
    return 0;
}
