#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// Simulate a fully-associative, `cache_lines`-capacity LRU cache, backed by
// a NEXT-LINE hardware prefetcher, over `line_trace[0..n)` (each entry is a
// cache LINE number -- already line-granularity, no byte offsets). Track
// the recency order explicitly as a list, most-recently-used at the FRONT,
// least-recently-used at the BACK (the eviction candidate).
//
// For each i in [0, n), in order:
//
//   1. DEMAND ACCESS to line_trace[i]:
//      - If it's already resident: a HIT. Remove it from its current spot
//        and re-insert it at the FRONT (MRU).
//      - Otherwise: a MISS. If the cache is full (size == cache_lines),
//        evict the line at the BACK (LRU) first. Insert line_trace[i] at
//        the FRONT (MRU).
//
//   2. NEXT-LINE PREFETCH for (line_trace[i] + 1), issued right after step
//      1 completes (so it sees any eviction step 1 just did):
//      - If that line is ALREADY resident: do nothing (no state change).
//      - Otherwise: if the cache is full, evict the line at the BACK
//        (LRU) first. Insert (line_trace[i] + 1) at the BACK -- a
//        prefetched-but-not-yet-demanded line starts life as the very
//        next eviction candidate.
//
// Prefetches never count toward hits or misses -- only demand accesses in
// step 1 do. Write the total demand hit and miss counts to *hits_out and
// *misses_out.
// ============================================================================
void simulate_next_line_prefetch(const int* line_trace, int n, int cache_lines,
                                  int* hits_out, int* misses_out);
