#include <cstdio>
#include <cstdint>
#include <set>
#include "sol.hpp"

// FIXED driver. Builds a 64-byte-aligned AoS array of Particle and an
// equivalent 64-byte-aligned SoA array holding only the x field, fills both
// with the same deterministic values, then sweeps the x field through each
// layout and prints the sum (must match) and the number of distinct 64-byte
// cache lines the sweep touched (measures the layout, not hardware).

static std::set<std::uintptr_t> g_lines;

void cache_reset() { g_lines.clear(); }

void touch(const void* p) {
    g_lines.insert(reinterpret_cast<std::uintptr_t>(p) / 64);
}

int lines_touched() { return static_cast<int>(g_lines.size()); }

constexpr int N = 1024;

alignas(64) static Particle g_aos[N];
alignas(64) static float g_soa_x[N];

int main() {
    for (int i = 0; i < N; ++i) {
        float v = static_cast<float>(i);
        g_aos[i] = Particle{v, v + 1.0f, v + 2.0f, i};
        g_soa_x[i] = v;
    }

    cache_reset();
    float sum_aos = sum_field_aos(g_aos, N);
    int lines_aos = lines_touched();

    cache_reset();
    float sum_soa = sum_field_soa(g_soa_x, N);
    int lines_soa = lines_touched();

    double ratio = lines_aos != 0 ? static_cast<double>(lines_soa) / lines_aos : 0.0;

    printf("%.6f %d\n", sum_aos, lines_aos);
    printf("%.6f %d\n", sum_soa, lines_soa);
    printf("ratio=%.6f\n", ratio);
    return 0;
}
