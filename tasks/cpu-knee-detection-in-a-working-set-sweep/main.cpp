#include <cstdio>
#include "sol.hpp"

// Deterministic per-sample jitter fraction (NOT randomness, no rand()/seed
// needed): alternates +-3% of whatever base latency it's applied to, so
// within-plateau samples are not perfectly flat. Because it SCALES with
// the base latency (as real measurement noise roughly does), its RAW
// (cycle) size varies a lot across levels -- a few tenths of a cycle at
// L1, several cycles at DRAM -- while its RELATIVE size stays a constant,
// small 3-6%, safely under every rel_threshold used below.
static double jitter_frac(int i) { return ((i % 2) == 0) ? 0.03 : -0.03; }

// FIXED driver, two fixture scenarios.
//
// Scenario 1: working-set size doubles from 1 KiB to 512 MiB across 20
// samples; latency is 4.0 cycles up to 32 KiB (L1), 12.0 up to 256 KiB
// (L2), 40.0 up to 8192 KiB (L3), else 180.0 (DRAM) -- 3 true knees, at
// the L1->L2, L2->L3 and L3->DRAM boundaries.
//
// Scenario 2: working-set size doubles from 1 KiB to 2048 KiB across 12
// samples; latency is 3.0 up to 16 KiB (L1), 10.0 up to 128 KiB (L2),
// else 150.0 (DRAM, no L3 modeled) -- 2 true knees.
int main() {
    // ---- scenario 1 ----
    {
        const int n = 20;
        double latency[20];
        long ws_kb = 1;
        for (int i = 0; i < n; i++) {
            double base;
            if (ws_kb <= 32) base = 4.0;
            else if (ws_kb <= 256) base = 12.0;
            else if (ws_kb <= 8192) base = 40.0;
            else base = 180.0;
            latency[i] = base * (1.0 + jitter_frac(i));
            ws_kb *= 2;
        }
        int idx[32];
        double threshold = 0.5;
        int count = detect_knees(latency, n, threshold, idx);
        printf("n=%d threshold=%.2f count=%d indices=", n, threshold, count);
        for (int i = 0; i < count; i++) printf("%d ", idx[i]);
        printf("\n");
    }

    // ---- scenario 2 ----
    {
        const int n = 12;
        double latency[12];
        long ws_kb = 1;
        for (int i = 0; i < n; i++) {
            double base;
            if (ws_kb <= 16) base = 3.0;
            else if (ws_kb <= 128) base = 10.0;
            else base = 150.0;
            latency[i] = base * (1.0 + jitter_frac(i));
            ws_kb *= 2;
        }
        int idx[32];
        double threshold = 0.3;
        int count = detect_knees(latency, n, threshold, idx);
        printf("n=%d threshold=%.2f count=%d indices=", n, threshold, count);
        for (int i = 0; i < count; i++) printf("%d ", idx[i]);
        printf("\n");
    }

    return 0;
}
