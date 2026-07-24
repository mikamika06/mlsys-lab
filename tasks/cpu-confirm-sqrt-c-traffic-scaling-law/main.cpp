#include <cstdio>
#include <cmath>
#include "sol.hpp"

// FIXED driver. Generates real (cache capacity, DRAM traffic) sample
// points from an actual cache-blocked N x N matmul run against a
// deterministic direct-mapped byte-address cache model -- both defined
// right here, not in solve.cpp/ref.cpp, so this part is identical no
// matter which sol.hpp implementation gets linked. The point of the
// task is the log-log regression in fit_scaling_exponent(), not this
// data-generation step.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int MAX_SETS = 16384 / LINE_BYTES;  // largest capacity we sweep / line size

long g_tag[MAX_SETS];
bool g_valid[MAX_SETS];
int g_sets;
long g_misses;

void cache_reset(long capacity_bytes) {
    g_sets = (int)(capacity_bytes / LINE_BYTES);
    for (int i = 0; i < g_sets; i++) g_valid[i] = false;
    g_misses = 0;
}

void cache_touch(long addr) {
    long line = addr / LINE_BYTES;
    int set = (int)(line % g_sets);
    if (g_valid[set] && g_tag[set] == line) return;  // hit
    g_misses++;
    g_valid[set] = true;
    g_tag[set] = line;
}

// Cache-blocked N x N x N matmul: block size B chosen so 3 B x B float
// tiles (A, B, C) fit in `capacity_bytes`. Returns total DRAM traffic in
// bytes (miss_count * LINE_BYTES).
long blocked_matmul_traffic(int N, long capacity_bytes) {
    int B = (int)std::sqrt((double)capacity_bytes / (3.0 * 4.0));
    if (B < 1) B = 1;
    if (B > N) B = N;
    long a_base = 0;
    long b_base = a_base + (long)N * N * 4;
    long c_base = b_base + (long)N * N * 4;

    cache_reset(capacity_bytes);
    for (int ii = 0; ii < N; ii += B) {
        int i_end = ii + B < N ? ii + B : N;
        for (int jj = 0; jj < N; jj += B) {
            int j_end = jj + B < N ? jj + B : N;
            for (int kk = 0; kk < N; kk += B) {
                int k_end = kk + B < N ? kk + B : N;
                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        for (int k = kk; k < k_end; k++) {
                            cache_touch(a_base + (long)(i * N + k) * 4);
                            cache_touch(b_base + (long)(k * N + j) * 4);
                            cache_touch(c_base + (long)(i * N + j) * 4);
                        }
                    }
                }
            }
        }
    }
    return g_misses * LINE_BYTES;
}
}  // namespace

int main() {
    const int N = 128;
    const int n_points = 5;
    double capacity[n_points] = {1024, 2048, 4096, 8192, 16384};
    double traffic[n_points];
    for (int i = 0; i < n_points; i++) {
        traffic[i] = (double)blocked_matmul_traffic(N, (long)capacity[i]);
    }

    double b = fit_scaling_exponent(capacity, traffic, n_points);

    for (int i = 0; i < n_points; i++) {
        printf("C=%.0f traffic=%.0f\n", capacity[i], traffic[i]);
    }
    printf("fitted_exponent=%.6f\n", b);
    return 0;
}
