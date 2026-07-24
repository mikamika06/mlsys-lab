#pragma once

// ============================================================================
// FIXED cache-line probe (defined in main.cpp — do not modify). A real
// set-associative LRU cache — 32 sets, 4 ways, 64-byte lines (8 KiB total,
// 1024 doubles) — pinned by the driver.
//   cache_reset() empties it.
//   touch(p) looks up/inserts the 64-byte line containing address p and
//     returns whether that access was a HIT.
//   miss_count() returns the number of misses recorded since the last
//     cache_reset().
// ============================================================================
void cache_reset();
bool touch(const void* p);
int miss_count();

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// sum_all_columns: sweep an R x C matrix COLUMN-MAJOR — outer loop over
// column j in [0, C), inner loop over row i in [0, R) — and return the sum
// of every element.
//
// You choose the matrix's leading dimension (row stride, in doubles)
// yourself — it does not have to equal C — and must allocate and fill the
// R*ld-double buffer inside this function using
//     value(i, j) = (i * 131 + j * 977) % 1009
// (a formula that depends only on i and j, never on ld, so every correct
// implementation returns the same sum no matter what ld it picks).
//
// Call touch(&data[i*ld + j]) for every element you read, in the exact
// order your sweep visits it — miss_count() after the call is graded.
// ============================================================================
double sum_all_columns(int R, int C);
