#pragma once

// Harness hook (declared here, DEFINED in main.cpp). Call this once for
// every byte address you access; the harness buckets addresses into
// 64-byte cache lines and reports how many DISTINCT lines were touched
// (repeated/overlapping touches are free — it's a set).
void touch(long byte_addr);

enum class Layout { AoS, SoA, AoSoA };

// Modeling convention for this task (packed, no alignment padding, so the
// byte math stays simple): N records, 4 fields per record with fixed
// byte sizes {4, 4, 4, 8} (record_size = 20 bytes). AoSoA groups records
// into blocks of 8, field-major within each block (block_size = 160
// bytes: a 32-byte sub-array per 4-byte field, a 64-byte sub-array for
// the 8-byte field).
//
// Call touch(addr) once for every byte address that would be read to
// satisfy this access, given where those bytes physically live under
// `layout`:
//   field_idx == -1   -> read every field of every one of the n records.
//   field_idx in [0,3] -> read only that field of every one of the n
//                          records.
void emit_access(Layout layout, int n, int field_idx);
