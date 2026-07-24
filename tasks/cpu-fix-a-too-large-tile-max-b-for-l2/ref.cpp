#include "sol.hpp"

int max_tile_b_for_l2(long l2_bytes) {
    long b = 0;
    while (3L * (b + 1) * (b + 1) * 4 <= l2_bytes) b++;
    return (int)b;
}
