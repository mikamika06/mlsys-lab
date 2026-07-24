#include <cstdio>
#include "sol.hpp"

static const int LINE_BYTES = 64;
static const int NUM_LINES = 256;   // 256 * 64B = 16KB direct-mapped cache

static long g_tag[NUM_LINES];
static bool g_valid[NUM_LINES];
static long g_misses = 0;

void reset_cache() {
    for (int i = 0; i < NUM_LINES; i++) g_valid[i] = false;
    g_misses = 0;
}

void touch_byte(long addr) {
    long line_num = addr / LINE_BYTES;
    int idx = (int)(line_num % NUM_LINES);
    long tag = line_num / NUM_LINES;
    if (!g_valid[idx] || g_tag[idx] != tag) {
        g_misses++;
        g_valid[idx] = true;
        g_tag[idx] = tag;
    }
}

long miss_count() { return g_misses; }

static void run_case(int seq_len, int head_dim) {
    const long Q_BASE = 0;
    const long K_BASE = 1000000;

    reset_cache();
    simulate_score_matrix_traffic(0, seq_len, head_dim, Q_BASE, K_BASE);
    long misses_rowmajor = miss_count();

    reset_cache();
    simulate_score_matrix_traffic(1, seq_len, head_dim, Q_BASE, K_BASE);
    long misses_transposed = miss_count();

    int choice = pick_better_layout(seq_len, head_dim);

    printf("S=%d D=%d rowmajor_misses=%ld transposed_misses=%ld choice=%d\n",
           seq_len, head_dim, misses_rowmajor, misses_transposed, choice);
}

int main() {
    run_case(16, 8);
    run_case(32, 16);
    run_case(64, 64);
    return 0;
}
