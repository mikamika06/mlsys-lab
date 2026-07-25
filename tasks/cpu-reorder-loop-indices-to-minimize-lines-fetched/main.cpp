#include <cstdio>
#include "sol.hpp"

// Deterministic direct-mapped cache model (harness code, not learner
// code): 64-byte lines, 8 sets -> 512 bytes total, 1 way per set.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 8;  // 512 bytes total, direct-mapped
long g_tag[NUM_LINES];
bool g_valid[NUM_LINES];
long g_misses;
}  // namespace

void reset_cache() {
    for (int i = 0; i < NUM_LINES; i++) g_valid[i] = false;
    g_misses = 0;
}

void touch_byte(long addr) {
    long line = addr / LINE_BYTES;
    int set = (int)(line % NUM_LINES);
    if (g_valid[set] && g_tag[set] == line) return;  // hit
    g_misses++;
    g_valid[set] = true;
    g_tag[set] = line;
}

long miss_count() { return g_misses; }

// FIXED driver. 64x64 matrix -> 16384 bytes of simulated address space,
// far bigger than the 512-byte cache, laid out row-major.
int main() {
    const int R = 64, C = 64;
    double values[R * C];
    for (int i = 0; i < R * C; i++) values[i] = (double)((i * 7 % 101) - 50);

    reset_cache();
    double sum = sum_matrix(values, 0, R, C);
    long misses = miss_count();

    printf("sum=%.1f misses=%ld\n", sum, misses);
    return 0;
}
