#pragma once

// `buf` holds n fixed-size records packed back-to-back: record i occupies
// buf[i*struct_size .. i*struct_size + struct_size). Each record has an
// 8-byte double payload starting at byte offset `field_offset` within it.
// Extract that double from every record, multiply it by 2.0, and write
// the n results into `out` (caller-allocated, n entries), in the same
// order as the records.
//
// A straight, branch-free, index-only scan like this — no data-dependent
// offsets, no reading a previous output to compute the next one — is
// exactly what lets a compiler autovectorize the loop. Deriving element
// i's address from anything other than i itself defeats that.
void optimize_vector_loop(const unsigned char* buf, int n, int struct_size, int field_offset, double* out);
