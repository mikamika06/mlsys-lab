#include <cstdio>
#include "sol.hpp"

// FIXED driver. line_bytes=64, W=4. The first 8 accesses bounce between
// two lines (0 and 64) -- a small, reused working set. The last 8
// accesses stream through 8 brand-new, never-repeated lines -- every
// window fully inside that region touches W=4 distinct lines, a bigger
// working set than anything in the bouncing region.
int main() {
    const int n = 16;
    long addrs[n] = {
        0, 64, 0, 64, 0, 64, 0, 64,               // reused: 2 lines
        128, 192, 256, 320, 384, 448, 512, 576,   // streaming: 8 distinct lines
    };
    const int line_bytes = 64;
    const int W = 4;

    long ws = max_working_set_bytes(addrs, n, line_bytes, W);
    printf("max_working_set_bytes=%ld\n", ws);
    return 0;
}
