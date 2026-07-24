#pragma once
#include <cstdint>

// Per-trace 3C miss classification counts.
struct MissCounts {
    int cold;
    int capacity;
    int conflict;
};

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// classify_misses: replay `addrs[0..n)` (byte addresses) through TWO LRU
// caches built from scratch inside this call (no state may persist across
// calls):
//   1. the REAL cache — set-associative, `sets` sets of `ways` ways each,
//      `line_bytes`-byte lines. set index = (addr / line_bytes) % sets.
//   2. a FULLY-ASSOCIATIVE cache with the SAME total capacity
//      (sets * ways lines, i.e. one set with sets*ways ways) — the cache
//      you would have if the same number of lines could hold ANY address
//      (no placement restriction).
//
// Classify every miss the real cache produces using the standard 3C model:
//   - cold:     this is the FIRST time this address's line has ever
//               appeared in the trace. Always a miss, independent of what
//               it evicts.
//   - capacity: not cold, and the fully-associative cache ALSO misses on
//               this access — the working set genuinely does not fit in
//               this many lines, no placement policy would help.
//   - conflict: not cold, and the fully-associative cache HITS on this
//               access — with unlimited associativity the line would still
//               be resident, so the miss is purely a placement/index
//               collision inside one set of the real cache.
//
// Both simulated caches use LRU eviction. Return the three counts (they sum
// to the real cache's total miss count; real-cache HITS are not counted at
// all).
// ============================================================================
MissCounts classify_misses(const uint64_t* addrs, int n,
                            int line_bytes, int sets, int ways);
