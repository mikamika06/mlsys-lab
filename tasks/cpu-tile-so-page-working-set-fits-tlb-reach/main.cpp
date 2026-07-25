#include <cstdio>
#include "sol.hpp"

// Deterministic fully-associative LRU TLB model (harness code, not
// learner code): 8 entries, 4096-byte pages. A tiny linear-scan LRU
// (fine for 8 entries) -- exact, deterministic, no hardware.
namespace {
constexpr int PAGE_BYTES = 4096;
constexpr int NUM_ENTRIES = 8;
long g_page[NUM_ENTRIES];
long g_last_used[NUM_ENTRIES];
bool g_valid[NUM_ENTRIES];
long g_clock;
long g_misses;
}  // namespace

void reset_tlb() {
    for (int i = 0; i < NUM_ENTRIES; i++) g_valid[i] = false;
    g_clock = 0;
    g_misses = 0;
}

void touch_page(long addr) {
    long page = addr / PAGE_BYTES;
    g_clock++;
    for (int i = 0; i < NUM_ENTRIES; i++) {
        if (g_valid[i] && g_page[i] == page) {
            g_last_used[i] = g_clock;  // hit
            return;
        }
    }
    g_misses++;
    // evict the least-recently-used entry (or first free slot)
    int victim = 0;
    long oldest = -1;
    for (int i = 0; i < NUM_ENTRIES; i++) {
        if (!g_valid[i]) { victim = i; break; }
        if (oldest == -1 || g_last_used[i] < oldest) { oldest = g_last_used[i]; victim = i; }
    }
    g_valid[victim] = true;
    g_page[victim] = page;
    g_last_used[victim] = g_clock;
}

long tlb_miss_count() { return g_misses; }

// FIXED driver. 64x256 matrix of doubles (131072 simulated bytes = 32
// pages), far bigger than the 32768-byte (8-page) TLB reach.
int main() {
    const int R = 64, C = 256;
    double values[R * C];
    for (int i = 0; i < R * C; i++) values[i] = (double)((i * 13 % 97) - 48);

    reset_tlb();
    double sum = sum_matrix_tlb_friendly(values, 0, R, C);
    long misses = tlb_miss_count();

    printf("sum=%.1f tlb_misses=%ld\n", sum, misses);
    return 0;
}
