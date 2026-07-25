#include "sol.hpp"

// TODO: zero C, then blocked triple loop over (ii,jj,kk) in steps of
// `tile`, inner triple loop over the (possibly partial) tile. See
// sol.hpp.
void tiled_matmul(const float* A, const float* B, float* C, int N, int tile) {
    (void)A; (void)B; (void)N; (void)tile;
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    // your code here
}
