#include <cstdio>
#include <cstdint>
#include <set>
#include "sol.hpp"

// FIXED driver. Builds a 64-byte-aligned base array of BASE_N floats, and
// two fixed, deterministic index arrays of length N: one clustered inside
// a single 64-byte line's worth of floats, one spread across the whole
// base array with a prime stride. Runs the same gather() over both and
// prints the checksum (must match a hand computable value) and the number
// of distinct 64-byte cache lines touched (measures the index pattern,
// not hardware).

static std::set<std::uintptr_t> g_lines;

void reset_lines() { g_lines.clear(); }

void touch(const void* p) {
    g_lines.insert(reinterpret_cast<std::uintptr_t>(p) / 64);
}

int lines_touched() { return static_cast<int>(g_lines.size()); }

constexpr int BASE_N = 2048;
constexpr int N = 200;

alignas(64) static float g_base[BASE_N];
static int g_idx_local[N];
static int g_idx_scattered[N];
static float g_result[N];

int main() {
    for (int i = 0; i < BASE_N; ++i) g_base[i] = static_cast<float>(i);

    // Clustered: every index falls in [0, 16) -- one 64-byte line's worth
    // of floats (16 floats * 4 bytes == 64 bytes).
    for (int i = 0; i < N; ++i) g_idx_local[i] = i % 16;

    // Scattered: prime stride 97 spreads indices across the whole base
    // array, touching a different line almost every time.
    for (int i = 0; i < N; ++i) g_idx_scattered[i] = (i * 97) % BASE_N;

    reset_lines();
    gather(g_base, g_idx_local, N, g_result);
    float sum_local = 0.0f;
    for (int i = 0; i < N; ++i) sum_local += g_result[i];
    int lines_local = lines_touched();

    reset_lines();
    gather(g_base, g_idx_scattered, N, g_result);
    float sum_scattered = 0.0f;
    for (int i = 0; i < N; ++i) sum_scattered += g_result[i];
    int lines_scattered = lines_touched();

    double ratio = lines_local != 0 ? static_cast<double>(lines_scattered) / lines_local : 0.0;

    printf("%.6f %d\n", sum_local, lines_local);
    printf("%.6f %d\n", sum_scattered, lines_scattered);
    printf("ratio=%.6f\n", ratio);
    return 0;
}
