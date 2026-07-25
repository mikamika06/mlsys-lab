#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver + trace generators (harness code, not learner code).
// Y[i][p] = sum_j A[i][j] * V[p][j], an N x N matrix A against P=4
// fixed "query" rows V (P is small and stays constant regardless of
// N -- V and Y are always tiny and never the bottleneck). Only A's
// byte addresses are recorded in the trace; the interesting reuse
// story is entirely about whether A gets re-swept from scratch.
namespace {
constexpr int P = 4;
constexpr int TILE = 8;

// NAIVE: p outermost, i middle, j inner. A doesn't depend on p, so
// EVERY one of the P passes re-sweeps the ENTIRE N x N matrix A from
// the top.
void naive_trace(int N, std::vector<long>& t) {
    for (int p = 0; p < P; p++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                t.push_back((long)(i * N + j) * 4);
}

// TILED: A is split into TILE x TILE blocks (jj outer, ii next); for
// each block, all P passes run back to back over just that block
// before moving to the next block. Every element of A is touched only
// within its own block's visit -- once a block is done, A never comes
// back to it.
void tiled_trace(int N, std::vector<long>& t) {
    for (int jj = 0; jj < N; jj += TILE)
        for (int ii = 0; ii < N; ii += TILE)
            for (int p = 0; p < P; p++)
                for (int i = ii; i < ii + TILE; i++)
                    for (int j = jj; j < jj + TILE; j++)
                        t.push_back((long)(i * N + j) * 4);
}
}  // namespace

int main() {
    const int sizes[] = {16, 32, 64, 128, 256};

    for (int N : sizes) {
        std::vector<long> nt, tt;
        naive_trace(N, nt);
        tiled_trace(N, tt);

        long naive_dist = max_reuse_distance(nt.data(), (int)nt.size());
        long tiled_dist = max_reuse_distance(tt.data(), (int)tt.size());

        bool naive_fits = naive_dist <= L2_LINE_BUDGET;
        bool tiled_fits = tiled_dist <= L2_LINE_BUDGET;

        printf("N=%d naive_dist=%ld naive_fits=%d tiled_dist=%ld tiled_fits=%d\n",
               N, naive_dist, naive_fits ? 1 : 0, tiled_dist, tiled_fits ? 1 : 0);
    }
    return 0;
}
