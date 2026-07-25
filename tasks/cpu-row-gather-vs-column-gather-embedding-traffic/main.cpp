#include <cstdio>
#include "sol.hpp"

// FIXED driver: gather 5 spread-out indices out of a 1000 x 32 float32
// table (line_bytes=64).
int main() {
    int idx[5] = {3, 700, 15, 999, 42};
    const int k = 5, V = 1000, D = 32, elem_bytes = 4, line_bytes = 64;

    long out[2] = {-1, -1};  // sentinel: an empty starter leaves this untouched
    gather_line_traffic(idx, k, V, D, elem_bytes, line_bytes, out);

    printf("row_major=%ld column_major=%ld\n", out[0], out[1]);
    return 0;
}
