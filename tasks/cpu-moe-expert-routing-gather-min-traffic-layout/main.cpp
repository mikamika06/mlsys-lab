#include <cstdio>
#include "sol.hpp"

// Deterministic direct-mapped cache model (harness code, not learner
// code): 64-byte lines, 4 lines -> 256 bytes total, 1 way per set.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 4;  // 256 bytes total: exactly one expert's footprint
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

// FIXED driver. E=4 experts, W=64 floats (256 bytes = 4 lines) each,
// T=32 tokens routed round-robin (expert_id[t] = t % E) -- the worst
// order for a naive scan, since consecutive tokens always switch
// experts.
int main() {
    const int E = 4;
    const int W = 64;
    const int T = 32;
    const long base = 0;

    double weights[E * W];
    for (int i = 0; i < E * W; i++) weights[i] = (double)((i * 11 % 53) - 26);

    int expert_id[T];
    for (int t = 0; t < T; t++) expert_id[t] = t % E;

    double out[T];
    for (int t = 0; t < T; t++) out[t] = -999.0;  // sentinel

    reset_cache();
    moe_gather(weights, expert_id, T, W, E, base, out);
    long misses = miss_count();

    printf("misses=%ld\n", misses);
    for (int t = 0; t < T; t++) printf("%.1f ", out[t]);
    printf("\n");
    return 0;
}
