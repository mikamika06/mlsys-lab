#pragma once
#include <array>

// Five fixed candidate layouts each place a per-thread 8-byte int64
// counter (owned by thread t in {0,1,2,3}) at these byte addresses:
//   layout 0: t * 8                     (packed, stride = 8 B)
//   layout 1: t * 64                    (stride = 1 line)
//   layout 2: t * 128                   (stride = 2 lines)
//   layout 3: t * 8 + 64 * (t % 2)      (alternating line offset)
//   layout 4: t * 16                    (stride = 16 B)
//
// A layout causes FALSE SHARING when two or more threads' counters land
// on the SAME cache line -- floor(a_i / line_bytes) == floor(a_j / line_bytes)
// for some i != j -- even though the counters themselves are logically
// independent (no data is actually shared, but the coherence protocol
// bounces the line between cores anyway).
//
// Return, for each of the 5 layouts in order, true if it false-shares
// under a cache with the given line size, false if every thread lands on
// its own line.
std::array<bool, 5> classify_layouts(long line_bytes);
