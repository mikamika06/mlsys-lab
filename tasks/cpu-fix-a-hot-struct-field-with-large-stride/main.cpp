#include <cstdio>
#include "sol.hpp"

// Deterministic direct-mapped cache model (harness code, not learner
// code): 64-byte lines, 64 sets -> 4096 bytes total. Real hardware cache
// timing is not reproducible across machines, so this model -- not the
// CPU's actual cache -- is the sole source of every miss count printed.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 64;  // 4096 bytes total, direct-mapped
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

// FIXED driver. 200 records, 256 bytes each (fat struct: hot field +
// cold padding), hot field at offset 16 -> 51200-byte footprint, far
// bigger than the 4096-byte cache model, so nothing can stay resident
// between separate passes.
int main() {
    const int n = 200;
    const int stride = 256;
    const int hot_offset = 16;
    const long base = 0;

    double out[3] = {-1.0, -1.0, -1.0};  // sentinel: an empty starter leaves this untouched
    reset_cache();
    hot_field_stats(base, stride, hot_offset, n, out);
    long misses = miss_count();

    printf("sum=%.4f min=%.4f max=%.4f misses=%ld\n", out[0], out[1], out[2], misses);
    return 0;
}
