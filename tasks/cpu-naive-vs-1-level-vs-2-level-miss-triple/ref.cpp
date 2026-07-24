#include "sol.hpp"

static void touch3(int N, long a_base, long b_base, long c_base, int i, int j, int k) {
    touch(addr(N, a_base, i, k));
    touch(addr(N, b_base, k, j));
    touch(addr(N, c_base, i, j));
}

static void naive(int N, long a_base, long b_base, long c_base) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                touch3(N, a_base, b_base, c_base, i, j, k);
}

static void tiled_1level(int N, int tile1, long a_base, long b_base, long c_base) {
    for (int ii = 0; ii < N; ii += tile1)
        for (int jj = 0; jj < N; jj += tile1)
            for (int kk = 0; kk < N; kk += tile1)
                for (int i = ii; i < ii + tile1; i++)
                    for (int j = jj; j < jj + tile1; j++)
                        for (int k = kk; k < kk + tile1; k++)
                            touch3(N, a_base, b_base, c_base, i, j, k);
}

static void tiled_2level(int N, int tile1, int tile2, long a_base, long b_base, long c_base) {
    for (int ii = 0; ii < N; ii += tile1)
        for (int jj = 0; jj < N; jj += tile1)
            for (int kk = 0; kk < N; kk += tile1)
                for (int i2 = ii; i2 < ii + tile1; i2 += tile2)
                    for (int j2 = jj; j2 < jj + tile1; j2 += tile2)
                        for (int k2 = kk; k2 < kk + tile1; k2 += tile2)
                            for (int i = i2; i < i2 + tile2; i++)
                                for (int j = j2; j < j2 + tile2; j++)
                                    for (int k = k2; k < k2 + tile2; k++)
                                        touch3(N, a_base, b_base, c_base, i, j, k);
}

void matmul_miss_triple(int N, int tile1, int tile2,
                         long a_base, long b_base, long c_base, long* out) {
    reset_cache();
    naive(N, a_base, b_base, c_base);
    out[0] = miss_count();

    reset_cache();
    tiled_1level(N, tile1, a_base, b_base, c_base);
    out[1] = miss_count();

    reset_cache();
    tiled_2level(N, tile1, tile2, a_base, b_base, c_base);
    out[2] = miss_count();
}
