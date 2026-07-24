#include "sol.hpp"

// BUG: the loop nest visits j (the contraction axis) OUTERMOST and b
// (the batch axis) INNERMOST. The math is still correct -- every
// (b, i, j) triple is still visited exactly once, so Y ends up right --
// but X[b][j] and Y[b][i] both get accessed with b varying fastest,
// which jumps a whole row's stride on every single step instead of
// scanning contiguously. Fix the axis order: see sol.hpp for which loop
// should be outermost / middle / innermost, and why.
float einsum_bij(int B, int I, int J,
                  long x_base, long w_base, long y_base,
                  const float* X, const float* W, float* Y) {
    for (int b = 0; b < B; b++)
        for (int i = 0; i < I; i++)
            Y[b * I + i] = 0.0f;

    for (int j = 0; j < J; j++) {
        for (int i = 0; i < I; i++) {
            for (int b = 0; b < B; b++) {
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
