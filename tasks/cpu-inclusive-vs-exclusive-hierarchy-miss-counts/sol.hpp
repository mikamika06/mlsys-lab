#pragma once

// Pinned hierarchy shape (fixed): line size, and how many fully-associative
// LRU lines each level holds.
constexpr int LINE_BYTES = 64;
constexpr int L1_WAYS = 4;
constexpr int L2_WAYS = 16;

// ============================================================================
// Run the SAME trace of `n` byte addresses (addrs[0..n)) through the SAME
// 2-level hierarchy (an L1_WAYS-line L1, an L2_WAYS-line L2, both
// fully-associative LRU, LINE_BYTES-byte lines) under two different
// inclusion policies, and write:
//   out2[0] = miss count under the INCLUSIVE policy
//   out2[1] = miss count under the EXCLUSIVE policy
// A "miss" is an access found in NEITHER L1 NOR L2 (has to go to memory).
//
// INCLUSIVE: every line resident in L1 must also be resident in L2 (L1's
// contents are always a subset of L2's). On a true miss, the line is
// inserted into BOTH L1 and L2. Inserting into L2 may evict L2's LRU line
// -- and if that evicted line is currently resident in L1 too, it must be
// invalidated (removed) from L1 as well ("back-invalidation"), even
// though L1's own LRU order had nothing to do with evicting it. A line
// found only in L2 (L1 miss, L2 hit) is pulled into L1 (may evict L1's
// LRU line, which needs no further action since it's still in L2).
//
// EXCLUSIVE: a line lives in AT MOST ONE of L1 or L2 at a time. On a true
// miss, the line is inserted into L1 only; if that evicts L1's LRU line,
// the evicted line is inserted into L2 instead of being discarded
// ("victim fill"), itself possibly evicting (and discarding) L2's own LRU
// line. A line found in L2 (L1 miss, L2 hit) is PROMOTED: removed from
// L2, inserted into L1, with the same victim-fill handling for whatever
// L1 evicts to make room.
// ============================================================================
void hierarchy_miss_counts(const long* addrs, int n, long* out2);
