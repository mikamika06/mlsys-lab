#pragma once
#include <cstdint>

enum class Op { Add, Sub, Mul };

// Elementwise saturating a[i] `op` b[i] for i in [0, n), writing results
// to out[i]. Must never trigger signed-overflow UB: do the arithmetic in
// a wider type (int64_t) where it cannot overflow, then clamp to
// [INT32_MIN, INT32_MAX] before narrowing to int32_t.
void saturating_arithmetic(const int32_t* a, const int32_t* b, int n, Op op, int32_t* out);
