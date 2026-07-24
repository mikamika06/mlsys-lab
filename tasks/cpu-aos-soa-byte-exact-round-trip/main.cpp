#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sol.hpp"

// FIXED driver: build a deterministic AoS array, round-trip it through
// SoA and back, and compare every reconstructed particle byte-for-byte
// against the original (memcmp -- no tolerance, no rounding excuse).
static uint32_t g_rng = 99u;
static uint32_t next_rand() {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) & 0xFFFFFFu;
}

int main() {
    const int N = 50;
    static Particle original[N];
    for (int i = 0; i < N; i++) {
        original[i].x = (float)((int)(next_rand() % 2001) - 1000) / 4.0f;
        original[i].y = (float)((int)(next_rand() % 2001) - 1000) / 4.0f;
        original[i].z = (float)((int)(next_rand() % 2001) - 1000) / 4.0f;
        original[i].id = (int)(next_rand() % 1000000);
    }

    static float xs[N], ys[N], zs[N];
    static int ids[N];
    aos_to_soa(original, N, xs, ys, zs, ids);

    static Particle reconstructed[N];
    soa_to_aos(xs, ys, zs, ids, N, reconstructed);

    int matches = 0;
    for (int i = 0; i < N; i++) {
        int ok = memcmp(&original[i], &reconstructed[i], sizeof(Particle)) == 0 ? 1 : 0;
        matches += ok;
        printf("%d %d\n", i + 1, ok);
    }
    printf("matches %d\n", matches);
    return 0;
}
