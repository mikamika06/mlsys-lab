#pragma once
#include <cstddef>
#include <cstdint>

// ============================================================================
// Fixed record types (real C++ — sizes/alignment/offsets come from the
// actual compiler, not a modeled ABI).
//
// Compact: int id; double value; char flag; — the double forces 4 bytes of
// padding after `id` (to land `value` on an 8-byte boundary) and 7 bytes of
// tail padding after `flag` (to round the struct up to its 8-byte
// alignment), giving sizeof(Compact) == 24.
//
// Padded: the same three fields plus an oversized `char pad[40]` tail —
// sizeof(Padded) == 64, exactly one cache line per record.
// ============================================================================
struct Compact {
    int id;
    double value;
    char flag;
};

struct Padded {
    int id;
    double value;
    char flag;
    char pad[40];
};

// ============================================================================
// Fake cache-line probe (FIXED — do not modify; defined in main.cpp).
// touch(p) records which 64-byte cache line the address `p` falls in, into a
// set that cache_reset() clears. lines_touched() returns how many DISTINCT
// lines have been recorded since the last reset. The driver 64-byte-aligns
// every array it hands you, so the line count depends only on which
// addresses your code actually visits.
// ============================================================================
void cache_reset();
void touch(const void* p);
int lines_touched();

// ============================================================================
// LEARNER implements these two sweeps in solve.cpp.
//
// sum_value_compact: sum the `value` field over an array of `n` Compact
// records. Call touch(&arr[i]) once per element — reading any field of
// arr[i] pulls in the cache line(s) covering that record.
//
// sum_value_padded: same sweep over an array of `n` Padded records. Call
// touch(&arr[i]) once per element.
// ============================================================================
double sum_value_compact(const Compact* arr, int n);
double sum_value_padded(const Padded* arr, int n);
