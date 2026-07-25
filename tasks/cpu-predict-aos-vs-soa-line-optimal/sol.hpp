#pragma once

// A record has `F` fields with byte sizes `field_bytes[0..F)`. An access
// pattern reads every record `r` in `[0, N)`, and for each field `f`
// where `mask[f]` is true, reads that field's `field_bytes[f]` bytes for
// record `r`.
//
// Under AoS, fields are packed back-to-back per record (record size =
// sum of field_bytes; field f of record r sits at byte offset
// r*record_bytes + sum(field_bytes[0..f))).
//
// Under SoA, each field gets its OWN contiguous array of N elements, at
// a fresh base address padded up to a whole number of 64-byte lines so
// no two fields' arrays ever share a line (element r of field f's array
// sits at soa_base[f] + r*field_bytes[f]).
//
// Count the number of DISTINCT 64-byte cache lines the access pattern
// touches under each layout (a field access that straddles a line
// boundary touches every line it overlaps). Return 1 if SoA's line
// count is <= AoS's (SoA is at least as good), else 0 (AoS strictly
// wins).
int soa_is_optimal(int N, int F, const int* field_bytes, const bool* mask);
