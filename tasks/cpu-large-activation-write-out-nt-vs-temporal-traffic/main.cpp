#include <cstdio>
#include "sol.hpp"

const int LINE_BYTES = 64;

namespace {
constexpr int NUM_LINES = 256;  // 256 sets * 64 bytes = 16384-byte direct-mapped cache
long g_tag[NUM_LINES];
bool g_valid[NUM_LINES];
}  // namespace

void reset_cache() {
    for (int i = 0; i < NUM_LINES; i++) g_valid[i] = false;
}

bool touch_byte(long addr) {
    long line = addr / LINE_BYTES;
    int set = (int)(line % NUM_LINES);
    if (g_valid[set] && g_tag[set] == line) {
        return false;  // hit
    }
    g_valid[set] = true;
    g_tag[set] = line;
    return true;  // miss
}

void nontemporal_store(long addr) {
    (void)addr;  // bypasses the cache model entirely: no residency, no eviction
}

// FIXED driver. h_bytes=4096 (64 lines) is a "hot" tensor that fits the
// 16384-byte cache easily; a_bytes=65536 (1024 lines) is a large
// write-once activation, 4x the cache's total capacity.
int main() {
    const long h_bytes = 4096;
    const long a_bytes = 65536;

    long temporal_bytes = modeled_dram_traffic(h_bytes, a_bytes, false);
    long nontemporal_bytes = modeled_dram_traffic(h_bytes, a_bytes, true);
    double ratio = (double)temporal_bytes / (double)nontemporal_bytes;

    printf("temporal_bytes=%ld nontemporal_bytes=%ld ratio=%.6f\n",
           temporal_bytes, nontemporal_bytes, ratio);
    return 0;
}
