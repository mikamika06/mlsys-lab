#include "sol.hpp"

void scale_unrolled(const float* in, int n, float s, float* out) {
    int i = 0;
    int main_end = (n / 4) * 4;
    for (; i < main_end; i += 4) {
        out[i + 0] = s * in[i + 0];
        out[i + 1] = s * in[i + 1];
        out[i + 2] = s * in[i + 2];
        out[i + 3] = s * in[i + 3];
    }
    // epilogue: handle the n % 4 leftover elements one at a time
    for (; i < n; i++) {
        out[i] = s * in[i];
    }
}
