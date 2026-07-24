#include "sol.hpp"

static void rec(int N, int r0, int c0, int n) {
    if (n <= 8) {
        for (int i = r0; i < r0 + n; i++) {
            for (int j = c0; j < c0 + n; j++) {
                touch(in_addr(N, i, j));
                touch(out_addr(N, j, i));
            }
        }
        return;
    }
    int h = n / 2;
    rec(N, r0,     c0,     h);
    rec(N, r0,     c0 + h, h);
    rec(N, r0 + h, c0,     h);
    rec(N, r0 + h, c0 + h, h);
}

void co_transpose(int N) {
    rec(N, 0, 0, N);
}
