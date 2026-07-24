#include <cstdio>
#include <vector>
#include "sol.hpp"

// PROVIDED. Deterministic value generator (no rand(), no clock), bounded
// so summed magnitudes stay moderate across the largest test length.
static float detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 7u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return ((float)(x % 4000) / 1000.0f) - 2.0f;  // roughly [-2, 2)
}

// FIXED driver. Do not edit. Six fixed-length cases (4, 4, 16, 16, 256,
// 128), deterministic a/b vectors, calls the learner's neon_dot, prints
// each result with high precision.
int main() {
    int lens[] = {4, 4, 16, 16, 256, 128};
    int seed_off = 0;

    for (int len : lens) {
        std::vector<float> a((size_t)len), b((size_t)len);
        for (int i = 0; i < len; ++i) {
            a[(size_t)i] = detval(seed_off + i);
            b[(size_t)i] = detval(seed_off + 500000 + i);
        }
        seed_off += 10000;

        float r = neon_dot(a, b);
        printf("len=%d dot=%.6f\n", len, r);
    }
    return 0;
}
