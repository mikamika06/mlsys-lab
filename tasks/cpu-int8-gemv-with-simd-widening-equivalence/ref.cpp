#include "sol.hpp"

// Correct widening GEMV: both the product and the running sum live in a
// 32-bit lane, mirroring what NEON vmull_s8/vmlal_s8 (or x86 VPMADDWD)
// do at the instruction level -- widen first, accumulate wide, only
// narrow (if ever) after the whole row is summed.
void gemv_i8(const int8_t* A, const int8_t* x, int32_t* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            int32_t prod = (int32_t)A[r * cols + c] * (int32_t)x[c];
            acc += prod;
        }
        y[r] = acc;
    }
}
