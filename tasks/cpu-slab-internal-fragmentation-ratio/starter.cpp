#include "sol.hpp"

// TODO: for each request, round up to the smallest size class >= it,
// average the allocated/requested ratios. See sol.hpp.
double slab_fragmentation_ratio(const int* size_classes, int num_classes, const int* requests, int n) {
    (void)size_classes; (void)num_classes; (void)requests; (void)n;
    // your code here
    return 0.0;
}
