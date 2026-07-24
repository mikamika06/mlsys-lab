#pragma once

// Pinned cache-line size in bytes (defined in main.cpp).
extern const int LINE_BYTES;

// LEARNER IMPLEMENTS.
//
// Simulate a set-associative LRU cache over the fixed byte-address trace
// addrs[0..n) and return the MISS COUNT.
//
//   line = addr / LINE_BYTES     (the fixed-size line this address falls in)
//   set  = line % num_sets       (which of the cache's num_sets sets that
//                                  line maps to)
//
// Each set independently holds up to `ways` resident lines. On a miss
// with a full set, evict that set's least-recently-used line. On a hit,
// that line becomes the most-recently-used line in its own set (a plain
// miss never touches any other set).
long lru_cache_misses(const long* addrs, int n, int num_sets, int ways);

// Run the SAME trace through three cache configurations that all share
// the same total capacity (num_sets * ways == 16 lines in every case)
// but differ in associativity, and write the three miss counts to
// out3[0..2] (via lru_cache_misses) in this order:
//   out3[0] = direct-mapped      (num_sets = 16, ways = 1)
//   out3[1] = 4-way associative  (num_sets = 4,  ways = 4)
//   out3[2] = fully associative  (num_sets = 1,  ways = 16)
void miss_triple(const long* addrs, int n, long* out3);
