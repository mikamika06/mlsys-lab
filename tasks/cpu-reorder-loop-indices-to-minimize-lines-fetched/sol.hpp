#pragma once

// Deterministic direct-mapped cache model (harness code, defined in
// main.cpp): 64-byte lines, 8 sets -> 512 bytes total. touch_byte(addr)
// simulates reading the 4-byte value at byte address `addr` through this
// cache and counts a MISS whenever that line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// An R x C matrix of doubles is stored ROW-MAJOR at byte address `base`:
// element (r, c) has its REAL value at values[r*C + c], and its
// SIMULATED address (what the cache model tracks) is
// base + (r*C + c) * 4 -- i.e. as if it were a 4-byte float32 array in
// the deployed model, even though `values` here holds real doubles.
//
// Visit every one of the R*C elements exactly once, touch_byte() its
// simulated address, and accumulate its real value into the sum you
// return. The loop order (which index is outer, which is inner) does
// not change the sum, but it changes how many touch_byte() calls MISS:
// visit elements in ROW-MAJOR order (r outer, c inner), matching how the
// matrix is actually laid out, so consecutive touches land in the same
// or the next cache line.
double sum_matrix(const double* values, long base, int R, int C);
