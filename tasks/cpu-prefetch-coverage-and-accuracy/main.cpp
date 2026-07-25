#include <cstdio>
#include "sol.hpp"

// FIXED driver. Everything in this file (the cache models AND the
// access trace) is harness code, identical for every candidate -- the
// only thing under test is compute_coverage_accuracy() itself.
namespace {
constexpr int LINE_BYTES = 64;
constexpr int NUM_LINES = 64;  // 4096 bytes, direct-mapped

struct Cache {
    long tag[NUM_LINES];
    bool valid[NUM_LINES];
    bool prefetched_unconsumed[NUM_LINES];  // only meaningful on the prefetch-enabled cache
    void reset() {
        for (int i = 0; i < NUM_LINES; i++) { valid[i] = false; prefetched_unconsumed[i] = false; }
    }
};

// touch() on a plain (no-prefetch) cache: returns true on hit.
bool touch_plain(Cache& c, long line) {
    int set = (int)(line % NUM_LINES);
    bool hit = c.valid[set] && c.tag[set] == line;
    c.valid[set] = true;
    c.tag[set] = line;
    return hit;
}

// touch() on the next-line-prefetch cache. On a miss, also prefetches
// line+1 (flagged "prefetched, not yet consumed"). On a hit to a line
// that was flagged prefetched-and-unconsumed, that prefetch just proved
// useful: count it once, then clear the flag.
void touch_prefetch(Cache& c, long line, long& misses, long& total_pf, long& useful_pf) {
    int set = (int)(line % NUM_LINES);
    bool hit = c.valid[set] && c.tag[set] == line;
    if (hit) {
        if (c.prefetched_unconsumed[set]) {
            useful_pf++;
            c.prefetched_unconsumed[set] = false;
        }
        return;
    }
    misses++;
    c.valid[set] = true;
    c.tag[set] = line;
    c.prefetched_unconsumed[set] = false;
    // issue the next-line prefetch
    long pf_line = line + 1;
    int pf_set = (int)(pf_line % NUM_LINES);
    total_pf++;
    c.valid[pf_set] = true;
    c.tag[pf_set] = pf_line;
    c.prefetched_unconsumed[pf_set] = true;
}
}  // namespace

int main() {
    // Trace: 16 sequential lines (0..15), then 10 lines strided by 2
    // (16, 18, 20, ..., 34) -- built with real address arithmetic, not
    // hardcoded results.
    const int n_seq = 16;
    const int n_strided = 10;
    long trace[n_seq + n_strided];
    for (int i = 0; i < n_seq; i++) trace[i] = i;
    for (int i = 0; i < n_strided; i++) trace[n_seq + i] = n_seq + 2 * i;
    const int n = n_seq + n_strided;

    Cache baseline, pf;
    baseline.reset();
    pf.reset();
    long baseline_misses = 0;
    for (int i = 0; i < n; i++) {
        if (!touch_plain(baseline, trace[i])) baseline_misses++;
    }

    long pf_misses = 0, total_pf = 0, useful_pf = 0;
    for (int i = 0; i < n; i++) {
        touch_prefetch(pf, trace[i], pf_misses, total_pf, useful_pf);
    }

    double coverage = -1.0, accuracy = -1.0;  // sentinel: an empty starter leaves this untouched
    compute_coverage_accuracy(baseline_misses, total_pf, useful_pf, &coverage, &accuracy);

    printf("baseline_misses=%ld total_prefetches=%ld useful_prefetches=%ld coverage=%.6f accuracy=%.6f\n",
           baseline_misses, total_pf, useful_pf, coverage, accuracy);
    return 0;
}
