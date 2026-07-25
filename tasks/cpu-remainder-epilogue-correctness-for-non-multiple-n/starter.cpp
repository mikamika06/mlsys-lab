#include "sol.hpp"

// BUG: the unrolled main loop only processes the largest multiple of 4
// <= n; there is no epilogue, so any leftover n % 4 elements are never
// written and stay whatever garbage/sentinel out[] already held. Fix by
// adding a scalar loop after the unrolled one to handle i in
// [main_end, n).
void scale_unrolled(const float* in, int n, float s, float* out) {
    int main_end = (n / 4) * 4;
    for (int i = 0; i < main_end; i += 4) {
        out[i + 0] = s * in[i + 0];
        out[i + 1] = s * in[i + 1];
        out[i + 2] = s * in[i + 2];
        out[i + 3] = s * in[i + 3];
    }
}
