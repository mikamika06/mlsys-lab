#include "sol.hpp"

// TODO: for every i, j, k in [0, N), touch_byte() the address of
// A[i][k], B[k][j], C[i][j] in that order. See sol.hpp.
void naive_matmul(int N, long a_base, long b_base, long c_base) {
    (void)N; (void)a_base; (void)b_base; (void)c_base;
    // your code here
}

// TODO: cache-oblivious recursive matmul. Split into quadrants and
// recurse (base case N <= 8: same triple-loop pattern as naive_matmul,
// restricted to that sub-block but addressed with the full-matrix
// stride). See sol.hpp for the exact quadrant formulas and required
// call order.
void recursive_matmul(int N, long a_base, long b_base, long c_base) {
    (void)N; (void)a_base; (void)b_base; (void)c_base;
    // your code here
}
