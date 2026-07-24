#include <cstdio>
#include "sol.hpp"

// FIXED driver. line_bytes=64. Addresses 0,4,8 all fall in line 0;
// 63 and 64 straddle the line-0/line-1 boundary; 200 and 264 land in
// distinct lines 3 and 4; 8 repeats line 0 again.
int main() {
    const int n = 8;
    long addrs[n] = {0, 4, 8, 63, 64, 200, 264, 8};
    const int line_bytes = 64;

    long count = count_distinct_lines(addrs, n, line_bytes);
    printf("distinct_lines=%ld\n", count);
    return 0;
}
