#pragma once

// ============================================================================
// FIXED page-touch probe (defined in main.cpp — do not modify). A real
// fully-associative LRU TLB with a pinned number of entries, each covering
// one 16384-byte page.
//   tlb_reset() empties it.
//   touch_page(p) looks up/inserts the page containing address p.
//   tlb_miss_count() returns the number of misses since the last reset.
// ============================================================================
void tlb_reset();
void touch_page(const void* p);
int tlb_miss_count();

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// sum_matrix_reordered: sum EVERY element of an R x C row-major matrix
// `data` (leading dimension `ld` doubles per row — `ld >= C`), touching
// each element EXACTLY ONCE via touch_page(&data[i*ld + j]), in whatever
// ORDER over the R*C (i, j) pairs you choose. The order never changes the
// sum -- it only changes which pages get revisited when, and therefore how
// many of those touches land in the TLB.
// ============================================================================
double sum_matrix_reordered(const double* data, int R, int C, int ld);
