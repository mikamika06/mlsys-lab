#include "sol.hpp"

// BUG: the running sum (and the product feeding it) is kept in a 16-bit
// lane, as if a SIMD accumulator register had been left at the width of
// a single per-lane product instead of widened for the reduction across
// the whole row. Individual products can already approach 16-bit range
// (int8 x int8 up to 128*128), and summing 32 of them per row blows well
// past it -- the accumulator wraps around mid-row instead of matching
// the true 32-bit sum.
void gemv_i8(const int8_t* A, const int8_t* x, int32_t* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        int16_t acc = 0;
        for (int c = 0; c < cols; c++) {
            int16_t prod = (int16_t)((int32_t)A[r * cols + c] * (int32_t)x[c]);
            acc = (int16_t)(acc + prod);
        }
        y[r] = acc;
    }
}
