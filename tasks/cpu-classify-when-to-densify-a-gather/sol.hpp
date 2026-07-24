#pragma once

// ============================================================================
// Deterministic set-associative LRU cache model, OWNED by main.cpp (see
// task.md for its fixed parameters: line size, set count, ways).
//
// touch(byte_addr): perform one access to the given byte address, updating
// the model's LRU state. Returns true iff it was a HIT, false iff it was a
// MISS (and the model installed a new line, evicting the LRU way on a
// conflict). DEFINED in main.cpp -- do not reimplement the cache itself.
//
// cache_reset(): invalidate every line, as if starting from a cold cache.
// ============================================================================
bool touch(long byte_addr);
void cache_reset();

// Base byte addresses of the two "arrays" a scenario indexes into.
//   ORIG_BASE    -- the original source array (element i lives at
//                    ORIG_BASE + i * elem_bytes).
//   SCRATCH_BASE -- a small contiguous scratch buffer a densify pass would
//                    compact the needed elements into (rank r lives at
//                    SCRATCH_BASE + r * elem_bytes).
constexpr long ORIG_BASE = 0;
constexpr long SCRATCH_BASE = 1L << 20;

enum Strategy { GATHER = 0, DENSIFY = 1 };

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// `indices[0..k)` is the sequence of element indices a workload requests,
// in order (repeats allowed). `elem_bytes` is the size of one element.
//
// Using cache_reset() / touch() against the shared deterministic cache
// model, simulate BOTH candidate strategies and return whichever produces
// FEWER total cache misses (GATHER wins on a tie):
//
//   GATHER  -- read indices[i] directly from ORIG_BASE, k times, in the
//              given order. Cost = number of misses over those k touches.
//
//   DENSIFY -- first touch every DISTINCT value in `indices`, exactly
//              once each, in ASCENDING order, at ORIG_BASE (the one-time
//              compaction pass that copies each needed source element
//              into a compact buffer). Then satisfy all k requests by
//              touching SCRATCH_BASE at the requested index's RANK (its
//              0-based position among the sorted distinct values), in
//              the given order. Cost = misses over compaction pass PLUS
//              misses over the k scratch reads.
//
// Call cache_reset() immediately before simulating each strategy, so the
// two costs are measured independently (each starts from a cold cache).
// ============================================================================
int classify_gather_strategy(const long* indices, int k, long elem_bytes);
