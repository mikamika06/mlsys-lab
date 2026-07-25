#include <cstdio>
#include <cstdint>
#include "sol.hpp"

// Deterministic byte generator (LCG, fixed seed) -- spans the full int8_t
// range so both very large-magnitude products and long rows of them show
// up, exactly the regime a too-narrow accumulator wraps around in. No
// rand(), no time, no hardware entropy.
static uint32_t lcg_state = 20260725u;
static int8_t next_i8() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (int8_t)((lcg_state >> 16) & 0xFFu);
}

int main() {
    const int rows = 8, cols = 32;
    static int8_t A[rows * cols];
    static int8_t x[cols];
    static int32_t y[rows];

    for (int i = 0; i < rows * cols; i++) A[i] = next_i8();
    for (int j = 0; j < cols; j++) x[j] = next_i8();
    for (int r = 0; r < rows; r++) y[r] = 0;

    gemv_i8(A, x, y, rows, cols);

    for (int r = 0; r < rows; r++) printf("%d ", y[r]);
    printf("\n");
    return 0;
}
