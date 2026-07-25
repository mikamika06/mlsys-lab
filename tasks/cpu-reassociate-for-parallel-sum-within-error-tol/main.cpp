#include <cstdio>
#include "sol.hpp"

// FIXED driver.
namespace {
int g_max_depth = 0;
} // namespace

void reset_critical_path() { g_max_depth = 0; }

int record_add(int depth_a, int depth_b) {
    int d = 1 + (depth_a > depth_b ? depth_a : depth_b);
    if (d > g_max_depth) g_max_depth = d;
    return d;
}

int critical_path_depth() { return g_max_depth; }

constexpr int N = 64;

int main() {
    static float values[N];
    for (int i = 0; i < N; ++i) {
        values[i] = 1.0f + 0.1f * static_cast<float>(i % 7);
    }

    reset_critical_path();
    float sum = parallel_sum(values, N);
    int depth = critical_path_depth();

    printf("sum=%.6f\n", sum);
    printf("critical_path_depth=%d\n", depth);
    return 0;
}
