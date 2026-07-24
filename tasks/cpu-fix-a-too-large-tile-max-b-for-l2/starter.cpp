#include <cmath>
#include "sol.hpp"

// BUG: this only accounts for ONE B x B tile fitting in L2 (l2_bytes / 4
// bytes-per-float), forgetting that THREE tiles -- A, B, and the C
// accumulator -- must all be resident at once. The returned B is too
// large: 3 tiles of that size blow past L2 and spill.
int max_tile_b_for_l2(long l2_bytes) {
    long b = (long)std::sqrt((double)l2_bytes / 4.0);
    return (int)b;
}
