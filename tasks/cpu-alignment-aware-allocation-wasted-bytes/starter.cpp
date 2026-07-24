#include "sol.hpp"

// TODO: bump-allocate through sizes[0..n) in order, rounding the current
// offset up to each request's alignment before placing it, and sum the
// padding bytes that rounding introduced.
long total_wasted_bytes(const int* sizes, const int* alignments, int n) {
    (void)sizes;
    (void)alignments;
    (void)n;
    // your code here
    return 0;
}
