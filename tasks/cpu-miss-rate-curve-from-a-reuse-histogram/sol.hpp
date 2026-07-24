#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// `hist[0..max_dist)` is a REUSE-DISTANCE histogram built from a real
// access trace: hist[d] is the number of accesses whose reuse distance
// (the number of DISTINCT other addresses referenced since that same
// address was last touched) was exactly `d`. `cold_misses` is the count
// of accesses that were the FIRST-EVER touch of their address (reuse
// distance = infinity -- they can never hit, at any cache size).
// `total_accesses == cold_misses + sum(hist)`.
//
// Under a fully-associative LRU cache holding `cache_size` lines, an
// access with reuse distance `d` is a HIT iff `d < cache_size` (its
// address is still within the top `cache_size` positions of the LRU
// stack) and a MISS iff `d >= cache_size`. A cold access is always a
// miss, regardless of `cache_size`.
//
// Return the miss RATE (a fraction in [0,1]) at the given `cache_size`:
// this is the "tail sum" of the histogram -- every access whose reuse
// distance is `>= cache_size`, plus every cold miss -- divided by the
// total number of accesses.
// ============================================================================
double miss_rate_at_cache_size(const long* hist, int max_dist, long cold_misses,
                                long total_accesses, int cache_size);
