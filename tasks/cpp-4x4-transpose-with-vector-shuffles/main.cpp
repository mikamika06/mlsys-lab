#include <cstdio>
#include <cmath>
#include "sol.hpp"

// Deterministic pseudo-random float generator (no rand(), no time): a fixed
// sin-hash squashed to roughly [-5, 5], seeded per-fixture.
static float pseudo_rand(int idx, int seed) {
    float x = sinf((float)idx * 12.9898f + (float)seed * 78.233f) * 43758.5453f;
    x -= floorf(x);              // fractional part, in [0, 1)
    return (x - 0.5f) * 10.0f;   // spread to roughly [-5, 5]
}

int main() {
    const int NT = 3;
    int seeds[NT] = {0, 42, 1337};

    for (int t = 0; t < NT; t++) {
        float in[16], out[16];
        if (t == 0) {
            for (int i = 0; i < 16; i++) in[i] = (float)i;   // 0..15 arange tile
        } else {
            for (int i = 0; i < 16; i++) in[i] = pseudo_rand(i, seeds[t]);
        }
        for (int i = 0; i < 16; i++) out[i] = -1.0f;  // sentinel: starter leaves this untouched

        transpose4x4(in, out);

        for (int i = 0; i < 16; i++) printf("%.6f ", out[i]);
        printf("\n");
    }
    return 0;
}
