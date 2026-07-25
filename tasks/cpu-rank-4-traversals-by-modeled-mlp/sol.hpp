#pragma once

// One outcome of the windowed memory-latency simulation below.
struct MlpResult {
    long long cycles;     // total cycles to service all n_misses misses
    long long mlp_x1000;  // average misses-in-flight, x1000, truncated to int
};

// Simulate a memory system that can have up to `window` cache misses
// outstanding (in flight) at once, each taking `latency` cycles to
// complete, servicing `n_misses` misses total.
//
// If `chained` is true, miss i cannot even be ISSUED until miss i-1 has
// FULLY COMPLETED. This models pointer chasing: the address of the next
// load is computed from the DATA returned by the previous one, so there
// is nothing to prefetch and no way to overlap them -- the outstanding
// window never fills past 1, no matter how large `window` is.
//
// If `chained` is false, every miss's address is known up front (it's
// computed from a loop counter, never from previously-loaded data), so
// a miss only needs a free slot in the `window`-wide outstanding-miss
// buffer: independent misses overlap freely, up to `window` at a time.
//
// Simulation rule: process misses 0..n_misses-1 in order. For miss i,
// let ready = (chained ? completion-time-of-miss(i-1) : 0). Assign it to
// whichever of the `window` slots frees up earliest; its issue time is
// max(ready, that slot's free time); it completes `latency` cycles
// later, and that slot's free time becomes this completion time.
//
// Return {cycles, mlp_x1000}:
//   cycles     = the completion time of the last-finishing miss
//   mlp_x1000  = (n_misses * latency * 1000) / cycles     (integer division)
// mlp_x1000 is the textbook definition of memory-level parallelism --
// total miss-service-time divided by elapsed wall-clock time, i.e. the
// time-averaged number of misses in flight at once -- scaled by 1000 so
// it prints and compares as an exact integer.
MlpResult simulate_mlp(int n_misses, int window, int latency, bool chained);
