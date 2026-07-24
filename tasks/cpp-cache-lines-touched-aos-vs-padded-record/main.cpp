#include <cstdio>
#include <cstdint>
#include <set>
#include "sol.hpp"

// FIXED driver. Builds a 64-byte-aligned array of Compact and a
// 64-byte-aligned array of Padded, fills both with the same deterministic
// `value` sequence, sweeps `value` through each layout and prints the sum
// (must match) and the number of distinct 64-byte cache lines touched
// (measures the layout, not hardware).

static std::set<std::uintptr_t> g_lines;

void cache_reset() { g_lines.clear(); }

void touch(const void* p) {
    g_lines.insert(reinterpret_cast<std::uintptr_t>(p) / 64);
}

int lines_touched() { return static_cast<int>(g_lines.size()); }

constexpr int N = 1000;

alignas(64) static Compact g_compact[N];
alignas(64) static Padded g_padded[N];

int main() {
    for (int i = 0; i < N; ++i) {
        double v = static_cast<double>(i);
        g_compact[i] = Compact{i, v, 0};
        g_padded[i] = Padded{i, v, 0, {}};
    }

    cache_reset();
    double sum_c = sum_value_compact(g_compact, N);
    int lines_c = lines_touched();

    cache_reset();
    double sum_p = sum_value_padded(g_padded, N);
    int lines_p = lines_touched();

    double ratio = lines_c != 0 ? static_cast<double>(lines_p) / lines_c : 0.0;

    printf("%.6f %d\n", sum_c, lines_c);
    printf("%.6f %d\n", sum_p, lines_p);
    printf("ratio=%.6f\n", ratio);
    return 0;
}
