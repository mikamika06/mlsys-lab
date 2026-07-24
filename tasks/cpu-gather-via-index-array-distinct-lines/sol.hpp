#pragma once

// ============================================================================
// Fake cache-line probe (FIXED — do not modify; defined in main.cpp).
// touch(p) records which 64-byte cache line the address `p` falls in, into
// a set that reset_lines() clears. lines_touched() returns how many
// DISTINCT lines have been recorded since the last reset. The driver
// 64-byte-aligns the base array it hands you, so the line count depends
// only on which indices your gather actually visits.
// ============================================================================
void reset_lines();
void touch(const void* p);
int lines_touched();

// ============================================================================
// Gather `n` elements from `base` through `idx`: for i in [0, n),
// result[i] = base[idx[i]]. Call touch(&base[idx[i]]) exactly once per
// element, in order (reading base[idx[i]] pulls in the cache line(s)
// covering that float).
// ============================================================================
void gather(const float* base, const int* idx, int n, float* result);
