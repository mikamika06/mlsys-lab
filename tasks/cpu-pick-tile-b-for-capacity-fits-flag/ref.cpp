#include "sol.hpp"
#include <cmath>

int derive_tile_b(long capacity_bytes, int elem_size) {
    if (elem_size <= 0 || capacity_bytes <= 0) return 0;
    double approx = std::sqrt(static_cast<double>(capacity_bytes) / (3.0 * elem_size));
    long b = static_cast<long>(approx);
    if (b < 0) b = 0;
    // Correct for floating-point error to find the true maximum integer B.
    while (b > 0 && 3L * b * b * elem_size > capacity_bytes) --b;
    while (3L * (b + 1) * (b + 1) * elem_size <= capacity_bytes) ++b;
    return static_cast<int>(b);
}
