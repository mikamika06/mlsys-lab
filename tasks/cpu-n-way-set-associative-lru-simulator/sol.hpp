#pragma once
#include <cstdint>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// count_misses: replay `addrs[0 .. n)` (byte addresses) through an N-way
// set-associative LRU cache built from scratch inside this call (no state
// persists across calls):
//   - line index  = addr / line_bytes
//   - set index   = line index % sets
//   - each set holds up to `ways` lines, evicting the Least Recently Used
//     one (across BOTH hits and misses -- a hit makes that line the most
//     recently used one in its set, same as a miss that inserts a new
//     line).
// Return the total number of MISSES over the whole trace.
// ============================================================================
int count_misses(const uint64_t* addrs, int n,
                  int line_bytes, int sets, int ways);
