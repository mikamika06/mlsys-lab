#include "sol.hpp"

// Reference: exact algebraic formula, using truncating integer division
// (matches C++'s `/` on non-negative operands).
long long unroll_overhead_saved(long long N, long long U) {
    return 2 * (N - N / U);
}
