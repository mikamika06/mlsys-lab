#include "sol.hpp"

void fma_vectorized(const float* a, const float* b, const float* c,
                     float* out, int n, int width) {
    for (int base = 0; base < n; base += width) {
        for (int lane = 0; lane < width; lane++) {
            int i = base + lane;
            out[i] = a[i] * b[i] + c[i];
        }
        op_tick();  // one vector instruction covers the whole chunk
    }
}
