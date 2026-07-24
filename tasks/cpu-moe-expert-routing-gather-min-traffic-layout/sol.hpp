#pragma once

// Deterministic direct-mapped cache model (harness code, defined in
// main.cpp): 64-byte lines, 4 lines -> 256 bytes total (exactly one
// expert's weight-vector footprint, see below). touch_byte(addr)
// simulates reading the 4-byte float32 value at simulated byte address
// `addr` through this cache and counts a MISS whenever that line wasn't
// already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// E experts, each with a W-element float32 weight vector, laid out back
// to back in SIMULATED address space starting at byte address `base`:
// weight w of expert e lives at simulated address base + (e*W + w)*4.
// (The `weights` array below holds the real numbers for the answer; the
// simulated addresses are only what the cache model tracks.)
//
// T tokens are routed to experts via expert_id[0..T) (expert_id[t] in
// [0, E)). For every token t, touch_byte() every one of expert_id[t]'s W
// simulated addresses (W calls per token, T*W total), and write
// out[t] = sum of weights[expert_id[t]*W .. +W) (a real value, read from
// the `weights` array the driver gives you).
//
// The order you visit the T tokens in never changes any out[] value, but
// it changes how many touch_byte() calls MISS: process tokens GROUPED BY
// EXPERT (every token routed to expert 0, then every token routed to
// expert 1, ...) so each expert's weight vector is fetched into cache
// once per expert and reused by every other token that shares it,
// instead of re-fetching a (fully evicted) weight vector on almost every
// token when tokens are handled in their original interleaved order.
void moe_gather(const double* weights, const int* expert_id, int T, int W, int E, long base, double* out);
