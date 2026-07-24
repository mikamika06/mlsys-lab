#include "sol.hpp"

// TODO: compute C := alpha * (A @ B) + beta * C in place, using a blocked
// (tiled) triple loop with tile size block_size. See task.md for the loop
// structure and how to handle a partial tile at the edge of a dimension.
void blocked_gemm(const float* A, const float* B, float* C,
                   int M, int N, int K, int block_size,
                   float alpha, float beta) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    (void)block_size; (void)alpha; (void)beta;
    // your code here
}
