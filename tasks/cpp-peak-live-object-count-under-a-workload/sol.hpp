#pragma once

// ============================================================================
// Fixed instrumented probe (FIXED — do not modify). Real construction and
// destruction update g_live and g_peak_live automatically — g_peak_live is
// the high-water mark of g_live ever observed.
// ============================================================================
inline int g_live = 0;
inline int g_peak_live = 0;

struct Probe {
    int x;
    double y;
    Probe() {
        ++g_live;
        if (g_live > g_peak_live) g_peak_live = g_live;
    }
    ~Probe() { --g_live; }
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Replay `n` events: for event i, `is_alloc[i]` true means "allocate an
// object with id `ids[i]`" (a genuine `new Probe()`, keyed by that id),
// false means "free the object with id `ids[i]`" (a genuine `delete` of
// whatever you allocated under that id, if anything is currently live under
// it). g_live/g_peak_live update themselves via Probe's own ctor/dtor above
// — you must actually construct and destroy real Probe objects, not just
// track counts by hand.
// ============================================================================
void run_workload(const int* ids, const bool* is_alloc, int n);
