#include <cstdio>
#include <cstdint>
#include <set>
#include "sol.hpp"

// FIXED driver.
static std::set<std::uintptr_t> g_dst_lines;

void reset_dst_lines() { g_dst_lines.clear(); }

void store_temporal(float* p, float v) {
    *p = v;
    g_dst_lines.insert(reinterpret_cast<std::uintptr_t>(p) / 64);
}

void store_nontemporal(float* p, float v) {
    *p = v; // reaches memory correctly -- just never registered in cache
}

int dst_lines_touched() { return static_cast<int>(g_dst_lines.size()); }

constexpr int N = 512;

alignas(64) static float g_src[N];
alignas(64) static float g_dst[N];

int main() {
    for (int i = 0; i < N; ++i) g_src[i] = static_cast<float>(i) * 0.5f;
    for (int i = 0; i < N; ++i) g_dst[i] = -1.0f; // sentinel, overwritten by a correct copy

    reset_dst_lines();
    stream_copy(g_src, g_dst, N);

    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += g_dst[i];
    int lines = dst_lines_touched();

    printf("checksum=%.6f\n", sum);
    printf("dst_lines_touched=%d\n", lines);
    return 0;
}
