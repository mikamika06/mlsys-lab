#pragma once

// ============================================================================
// FIXED TLB probe (defined in main.cpp — do not modify). A fully-associative
// LRU TLB with a pinned number of entries (16).
//   tlb_reset(page_bytes)  empties the TLB and fixes the page size used by
//                          every touch_addr() call until the next reset.
//   touch_addr(addr)       looks up/inserts the page containing byte
//                          address `addr` (page = addr / page_bytes).
//   tlb_miss_count()       returns the number of misses since the last reset.
// ============================================================================
void tlb_reset(long page_bytes);
void touch_addr(long byte_addr);
int tlb_miss_count();

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// `indices[0..n)` is the sequence of embedding row indices an embedding
// gather accesses, in order. Row `r` starts at byte address `r * row_bytes`
// (only the base address of each row is touched — one TLB probe per
// accessed row).
//
// For EACH candidate page size `page_sizes[0..p)`:
//   1. tlb_reset(page_sizes[k])
//   2. touch_addr(indices[i] * row_bytes) for every i in [0, n), IN ORDER
//   3. read tlb_miss_count()
//
// Return the page size (the VALUE from page_sizes, not its index) whose
// run produced the FEWEST misses. Break ties by returning the SMALLER
// page size.
// ============================================================================
long choose_page_size(const int* indices, int n, int row_bytes, const long* page_sizes, int p);
