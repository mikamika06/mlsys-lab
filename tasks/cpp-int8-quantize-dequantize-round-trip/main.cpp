// Fixed driver: a small deterministic xorshift32 PRNG (fixed seed, pure
// integer arithmetic — no rand(), no clock) generates 500 floats in
// [-10, 10], which are quantized+dequantized and printed.
#include "sol.hpp"
#include <cstdint>
#include <cstdio>

int main() {
    const int N = 500;
    static float data[N];

    uint32_t state = 88172645u; // fixed seed
    for (int i = 0; i < N; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        double u = static_cast<double>(state % 2000001u) / 100000.0; // [0, 20]
        data[i] = static_cast<float>(u - 10.0);
    }

    static float out[N];
    quantize_dequantize(data, N, 0.1f, 5, out);

    for (int i = 0; i < N; i++) {
        printf("%.9f\n", out[i]);
    }
    return 0;
}
