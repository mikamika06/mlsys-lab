#include <cstdio>
#include "sol.hpp"

// FIXED driver: one (M,N,K) shape swept across four tile sizes (showing AI
// rise from near-memory-bound at tile=1 toward compute-bound as tile
// grows), plus two more shapes/element-sizes for coverage.
int main() {
    struct Case { long M, N, K, tile, elem_bytes; };
    Case cases[6] = {
        {256, 256, 256, 1, 4},
        {256, 256, 256, 8, 4},
        {256, 256, 256, 32, 4},
        {256, 256, 256, 128, 4},
        {1024, 1024, 64, 16, 4},
        {64, 64, 4096, 32, 8},
    };

    for (const auto& c : cases) {
        double ai = gemm_arithmetic_intensity(c.M, c.N, c.K, c.tile, c.elem_bytes);
        printf("%.9f\n", ai);
    }
    return 0;
}
