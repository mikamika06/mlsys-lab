#include <cstdio>
#include "sol.hpp"

// Deterministic direct-mapped cache models (harness code, not learner
// code): 64-byte lines, 16 sets -> 1024 bytes total, 1 way per set,
// each of the 3 policies with its own private state.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 16;

struct PlainCache {
    long tag[NUM_LINES];
    bool valid[NUM_LINES];
    long misses;
    void reset() {
        for (int i = 0; i < NUM_LINES; i++) valid[i] = false;
        misses = 0;
    }
    // fill() marks a line resident WITHOUT counting a miss (used for
    // speculative prefetches). Returns true if it was already resident.
    bool fill(long line) {
        int set = (int)(line % NUM_LINES);
        bool hit = valid[set] && tag[set] == line;
        valid[set] = true;
        tag[set] = line;
        return hit;
    }
};

PlainCache g_no_pf, g_next_line, g_stride;
long g_stride_last_addr;
long g_stride_last_delta;
bool g_stride_have_last;
}  // namespace

void reset_prefetch_caches() {
    g_no_pf.reset();
    g_next_line.reset();
    g_stride.reset();
    g_stride_last_addr = 0;
    g_stride_last_delta = 0;
    g_stride_have_last = false;
}

void touch_no_prefetch(long addr) {
    long line = addr / LINE_BYTES;
    if (!g_no_pf.fill(line)) g_no_pf.misses++;
}

void touch_next_line(long addr) {
    long line = addr / LINE_BYTES;
    if (!g_next_line.fill(line)) {
        g_next_line.misses++;
        g_next_line.fill(line + 1);  // free speculative prefetch, not counted
    }
}

void touch_stride(long addr) {
    long line = addr / LINE_BYTES;
    if (!g_stride.fill(line)) g_stride.misses++;

    if (g_stride_have_last) {
        long delta = addr - g_stride_last_addr;
        if (delta == g_stride_last_delta) {
            long next_line_pred = (addr + delta) / LINE_BYTES;
            g_stride.fill(next_line_pred);  // free speculative prefetch, not counted
        }
        g_stride_last_delta = delta;
    } else {
        g_stride_have_last = true;
    }
    g_stride_last_addr = addr;
}

long miss_count_no_prefetch() { return g_no_pf.misses; }
long miss_count_next_line() { return g_next_line.misses; }
long miss_count_stride() { return g_stride.misses; }

// FIXED driver. 40 steps, stride 128 bytes (2 lines) -- big enough that
// next-line prefetch (which only ever brings in the ADJACENT line) never
// prefetches an address the trace will actually touch.
int main() {
    const long base = 0;
    const int stride_bytes = 128;
    const int n_steps = 40;

    long out[3] = {-1, -1, -1};  // sentinel: an empty starter leaves this untouched
    reset_prefetch_caches();
    generate_and_run(base, stride_bytes, n_steps, out);

    printf("no_prefetch=%ld next_line=%ld stride=%ld\n", out[0], out[1], out[2]);
    return 0;
}
