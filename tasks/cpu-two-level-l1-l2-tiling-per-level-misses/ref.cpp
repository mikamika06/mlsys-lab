#include <algorithm>
#include <cstddef>
#include "sol.hpp"

void matmul_two_level_tiled(const double* A, const double* B, double* C,
                             int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int kL2Tile = 32;
    constexpr int kL1Tile = 8;

    for (int i2 = 0; i2 < M; i2 += kL2Tile) {
        int i2end = std::min(i2 + kL2Tile, M);
        for (int j2 = 0; j2 < N; j2 += kL2Tile) {
            int j2end = std::min(j2 + kL2Tile, N);

            for (int i1 = i2; i1 < i2end; i1 += kL1Tile) {
                int i1end = std::min(i1 + kL1Tile, i2end);
                for (int j1 = j2; j1 < j2end; j1 += kL1Tile) {
                    int j1end = std::min(j1 + kL1Tile, j2end);

                    // Local accumulator for this L1 tile (assumed to live
                    // in registers, never touch()ed) -- k is the OUTERMOST
                    // loop inside the tile, i/j innermost, so a fixed k
                    // sweeps every (i, j) in the tile together: A[i, k]
                    // is reused across all j in the tile (broadcast), and
                    // consecutive k's revisit the SAME handful of A/B
                    // cache lines before moving on.
                    double tile_acc[kL1Tile][kL1Tile] = {};
                    for (int k = 0; k < K; ++k) {
                        for (int i = i1; i < i1end; ++i) {
                            const double* ap = &A[static_cast<size_t>(i) * lda + k];
                            touch(ap);
                            double a_ik = *ap;
                            for (int j = j1; j < j1end; ++j) {
                                const double* bp = &B[static_cast<size_t>(k) * ldb + j];
                                touch(bp);
                                tile_acc[i - i1][j - j1] += a_ik * (*bp);
                            }
                        }
                    }
                    for (int i = i1; i < i1end; ++i)
                        for (int j = j1; j < j1end; ++j)
                            C[static_cast<size_t>(i) * ldc + j] = tile_acc[i - i1][j - j1];
                }
            }
        }
    }
}
