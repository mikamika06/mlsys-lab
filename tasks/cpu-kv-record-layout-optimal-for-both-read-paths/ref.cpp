#include "sol.hpp"

long thkd_addr(long base, int T, int H, int D, int E, int t, int h, int k, int d) {
    (void)T;
    long index = (((long)t * H + h) * 2 + k) * D + d;
    return base + index * E;
}
