#pragma once
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Compute row-wise dot products of two int8 matrices A, B (each M rows x
// N columns, N a multiple of 16, stored row-major in a flat buffer) into
// int32 accumulators, using REAL ARM NEON widening intrinsics
// (<arm_neon.h>) -- exactly the building block real int8 quantized
// inference kernels use:
//
//   for each row m:
//     res[m] = sum_{n=0}^{N-1} A[m*N+n] * B[m*N+n]     (as int32)
//
// int8 * int8 can reach +-16384, and summing 16+ of those can overflow a
// 16-bit accumulator, so widen BEFORE multiplying: load 16 int8 lanes
// with vld1q_s8, split with vget_low_s8/vget_high_s8, widening-multiply
// each half with vmull_s8 (int8x8 x int8x8 -> int16x8), and fold into a
// 32-bit accumulator with vpadalq_s16 (pairwise widening accumulate,
// int16x8 -> int32x4) so every partial product is safely represented
// before it is summed. Do not just write a plain scalar loop over
// int32_t -- that defeats the point of the exercise, even though it
// would (slowly) produce the same numbers.
// ---------------------------------------------------------------------------
std::vector<int32_t> int8_widening_dot_product(const std::vector<int8_t>& A,
                                                 const std::vector<int8_t>& B,
                                                 int M, int N);
