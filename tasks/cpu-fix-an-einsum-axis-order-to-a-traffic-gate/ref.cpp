#include "sol.hpp"

// Fixed: b outer, i middle, j (the contraction axis) innermost -- both
// X[b][j] and W[i][j] are scanned contiguously along their fast axis j,
// and X's row (b fixed for the whole i,j sub-loop) stays cache-resident
// across all I reuses.
float einsum_bij(int B, int I, int J,
                  long x_base, long w_base, long y_base,
                  const float* X, const float* W, float* Y) {
    for (int b = 0; b < B; b++)
        for (int i = 0; i < I; i++)
            Y[b * I + i] = 0.0f;

    for (int b = 0; b < B; b++) {
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < J; j++) {
                touch(mat_addr(x_base, J, b, j));
                touch(mat_addr(w_base, J, i, j));
                touch(mat_addr(y_base, I, b, i));
                Y[b * I + i] += X[b * J + j] * W[i * J + j];
            }
        }
    }

    double checksum = 0.0;
    for (int k = 0; k < B * I; k++) checksum += Y[k];
    return (float)checksum;
}
