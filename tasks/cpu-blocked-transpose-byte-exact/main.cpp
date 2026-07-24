#include <cstdio>
#include "sol.hpp"

// FIXED driver: a 12x12 matrix of doubles, transposed with block=4 (a 3x3
// grid of 4x4 tiles). Prints the full output matrix, one row per line.

constexpr int N = 12;
constexpr int BLOCK = 4;

int main() {
    double in[N * N];
    double out[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            in[i * N + j] = static_cast<double>(i * N + j) * 0.5;
        }
    }
    for (int k = 0; k < N * N; ++k) out[k] = -1.0;

    blocked_transpose(in, out, N, BLOCK);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.3f ", out[i * N + j]);
        }
        printf("\n");
    }
    return 0;
}
