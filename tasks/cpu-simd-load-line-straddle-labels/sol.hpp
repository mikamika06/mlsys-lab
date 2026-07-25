#pragma once
#include <cstdint>

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// straddles_line: a SIMD load reads `width_bytes` contiguous bytes starting
// at `base_addr` (a scalar double load is width 8, SSE xmm is 16, AVX ymm
// is 32, AVX-512 zmm is 64). Cache lines are `line_bytes`-aligned, fixed
// size chunks of memory. The load STRADDLES a line boundary if the byte
// range [base_addr, base_addr + width_bytes) is not entirely contained in
// one line -- i.e. it needs two separate cache-line fetches instead of one.
//
// Return true if the load straddles a line boundary, false if it stays
// within a single line (an aligned load that fits exactly, ending exactly
// on the boundary, does NOT straddle -- the boundary byte itself is never
// read).
// ============================================================================
bool straddles_line(uint64_t base_addr, int width_bytes, int line_bytes);
