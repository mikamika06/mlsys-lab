#pragma once

// ============================================================================
// Deterministic set-associative LRU cache model, OWNED by main.cpp (see
// task.md for its fixed parameters). touch(byte_addr) performs one access,
// updates LRU state, and returns true iff it was a HIT. cache_reset()
// invalidates every line, as if starting cold. DEFINED in main.cpp -- do
// not reimplement the cache model itself.
// ============================================================================
bool touch(long byte_addr);
void cache_reset();

// ============================================================================
// LEARNER implements both of these in solve.cpp.
//
// A decode step re-scans the WHOLE KV cache built up so far: record `t`
// (one token's combined K+V footprint, `rec_bytes` bytes) lives at byte
// address `t * rec_bytes`, for `t` in `[0, T)`, laid out back-to-back.
//
// simulate_decode_pass(T, rec_bytes, prefetch_distance):
//   Call cache_reset(), then for t = 0 .. T-1, in order:
//     1. DEMAND read of record t: touch(t * rec_bytes). Count a miss if
//        touch() returns false.
//     2. SOFTWARE PREFETCH: if prefetch_distance > 0 and
//        (t + prefetch_distance) < T, touch((t + prefetch_distance) *
//        rec_bytes) to warm the cache for a future record. This touch
//        does NOT count toward the miss total either way.
//   Return the total number of demand misses (step 1's only).
//
// choose_best_prefetch_distance(T, rec_bytes, max_distance):
//   Try every prefetch_distance in [0, max_distance] (inclusive), calling
//   simulate_decode_pass(T, rec_bytes, prefetch_distance) for each, and
//   return whichever prefetch_distance produced the FEWEST total misses
//   (the SMALLEST such distance on a tie).
//
// Too small a distance (0, or 1 when consecutive records already share a
// line) leaves demand reads stalling on cold data; too large a distance
// warms a record long before it's needed, and it gets evicted by the
// cache's own limited capacity before the demand read ever arrives.
// ============================================================================
long simulate_decode_pass(int T, int rec_bytes, int prefetch_distance);
int choose_best_prefetch_distance(int T, int rec_bytes, int max_distance);
