#pragma once

// Deterministic direct-mapped L2 model (harness code, defined in
// main.cpp): 64-byte lines, 64 sets -> 4096 bytes total. touch_byte(addr)
// simulates reading the 4-byte value at byte address `addr` through this
// cache and counts a MISS whenever that line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// A "tile" of side length B is 3 contiguous B x B float arrays (think:
// the A, B, C sub-blocks of a blocked matmul's inner loop), occupying
// 3*B*B*4 bytes starting at address 0: array f's element (r, c) is at
// byte address (f*B*B + r*B + c) * 4, for f in {0,1,2}, r,c in [0, B).
//
// pick_resident_tile: for each candidate tile_b0 and tile_b1, reset the
// cache and touch every address of that tile's 3 arrays, in order,
// `passes` times in a row (same addresses each pass, back to back --
// this models an inner k-loop reusing the same tile). Record the TOTAL
// miss count over all `passes` repetitions into out_misses[0] (for
// tile_b0) and out_misses[1] (for tile_b1). A tile whose footprint fits
// the 4096-byte L2 stays resident after the first pass, so only its
// first pass misses; a tile too big for L2 re-misses every pass.
//
// Return the id (0 or 1) of whichever tile produced the FEWER total
// misses -- the one that stayed resident in L2.
int pick_resident_tile(int tile_b0, int tile_b1, int passes, long* out_misses);
