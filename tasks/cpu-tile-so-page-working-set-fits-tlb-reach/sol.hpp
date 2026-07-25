#pragma once

// Deterministic fully-associative LRU TLB model (harness code, defined
// in main.cpp): 8 entries, 4096-byte pages -> "TLB reach" of
// 8 * 4096 = 32768 bytes. touch_page(addr) simulates translating the
// page containing byte address `addr` and counts a MISS whenever that
// page wasn't already resident (the least-recently-used entry is
// evicted when the TLB is full).
void reset_tlb();
void touch_page(long addr);
long tlb_miss_count();

// An R x C matrix of doubles is stored row-major at byte address `base`:
// element (r, c) has its real value at values[r*C + c], and its
// simulated address is base + (r*C + c) * 8.
//
// Visit every one of the R*C elements exactly once, touch_page() its
// simulated address, and accumulate its real value into the sum you
// return. The visiting order does not change the sum, but it changes
// how many touch_page() calls MISS: visit in ROW-MAJOR order (r outer, c
// inner), so the set of pages in flight at any moment stays within the
// TLB's 32768-byte reach, instead of a column-major sweep that revisits
// every page of the whole matrix once per column.
double sum_matrix_tlb_friendly(const double* values, long base, int R, int C);
