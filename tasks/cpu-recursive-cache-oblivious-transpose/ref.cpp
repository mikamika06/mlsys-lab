#include "sol.hpp"

static void rec(const float* in, float* out, int N, int r0, int c0, int n) {
    if (n <= 8) {
        for (int row = r0; row < r0 + n; row++) {
            for (int col = c0; col < c0 + n; col++) {
                out[col * N + row] = in[row * N + col];
                touch(in_addr(N, row, col));
                touch(out_addr(N, col, row));
            }
        }
        return;
    }
    int h = n / 2;
    rec(in, out, N, r0,     c0,     h);
    rec(in, out, N, r0,     c0 + h, h);
    rec(in, out, N, r0 + h, c0,     h);
    rec(in, out, N, r0 + h, c0 + h, h);
}

void co_transpose(const float* in, float* out, int N) {
    rec(in, out, N, 0, 0, N);
}
