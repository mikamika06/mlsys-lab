#pragma once

struct HitMiss {
    long hits;
    long misses;
};

// LEARNER IMPLEMENTS.
//
// Simulate a single-level, FULLY-ASSOCIATIVE LRU cache over the byte-
// address trace addrs[0..n). The cache holds `capacity` lines of
// `line_bytes` bytes each -- fully-associative means any resident line
// can occupy any slot, there is only one set.
//
//   line = addr / line_bytes
//
// For each access, in order:
//   - if `line` is already resident: HIT. That line becomes the
//     most-recently-used (every other resident line's relative
//     recency is unchanged).
//   - otherwise: MISS. Insert `line` as the most-recently-used. If the
//     cache was already at `capacity` lines before this insert, first
//     evict whichever resident line is currently the LEAST-recently-
//     used.
//
// Return the total {hits, misses} over the whole trace.
HitMiss simulate_lru(const long* addrs, int n, int capacity, int line_bytes);
