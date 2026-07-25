#include <cstdio>
#include "sol.hpp"

// FIXED driver. N=7 (not a multiple of tile=3, so the last block in
// each dimension is a partial 1x1/1x3/3x1 tile) -- exercises the
// boundary handling.
int main() {
    const int N = 7;
    const int tile = 3;
    float A[N * N], B[N * N], C[N * N];
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)((i % 5) - 2) * 0.5f;
        B[i] = (float)((i % 7) - 3) * 0.25f;
        C[i] = -999.0f;  // sentinel
    }

    tiled_matmul(A, B, C, N, tile);

    for (int i = 0; i < N * N; i++) printf("%.5f ", C[i]);
    printf("\n");
    return 0;
}
