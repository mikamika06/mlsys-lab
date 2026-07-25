#include "sol.hpp"

// TODO: for each candidate in page_sizes[0..p), tlb_reset(candidate), then
// touch_addr(indices[i] * row_bytes) for every i in [0, n) in order, then
// read tlb_miss_count(). Return the candidate with the fewest misses,
// breaking ties by the smaller page size. See sol.hpp.
long choose_page_size(const int* indices, int n, int row_bytes, const long* page_sizes, int p) {
    (void)indices; (void)n; (void)row_bytes; (void)page_sizes; (void)p;
    // your code here
    return 0;
}
