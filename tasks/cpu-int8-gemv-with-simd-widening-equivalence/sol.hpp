#pragma once
#include <cstdint>

// Quantized (int8) GEMV: y[r] = sum_{c=0}^{cols-1} A[r*cols+c] * x[c].
//
// Real SIMD widening instructions (NEON vmull_s8/vmlal_s8, x86 VPMADDWD)
// widen the int8 x int8 product AND the running sum they accumulate it
// into, to a lane wide enough that it cannot wrap around mid-row. This
// function must reproduce that widening exactly: every product and every
// partial sum lives in a 32-bit lane, never anything narrower.
void gemv_i8(const int8_t* A, const int8_t* x, int32_t* y, int rows, int cols);
