#pragma once

// Cache access hook, DEFINED in main.cpp: touch() records the byte
// address against a real (small, fixed) set-associative LRU cache AND
// bumps a running touch counter, hit or miss. Call touch() once per byte
// address you genuinely read or write -- never more than once for the
// same logical access, and never for bytes you don't actually need.
void touch(long byte_addr);
long touch_count();   // total touch() calls since the last reset_stats()
long miss_count();    // total cache misses among those touches
void reset_stats();   // zero both counters and start from a fresh cache

// `record_count` records are stored Array-of-Structs: record i has
// `field_count` 4-byte float fields laid out contiguously, starting at
// byte address `aos_base + i * field_count * 4`. Field `field_index`
// (0-based) of record i therefore lives at
//   aos_base + i * field_count * 4 + field_index * 4
//
// Transform ONLY that one field, for every record, into a fresh
// contiguous Struct-of-Arrays output array (4 bytes per element) starting
// at `soa_out_base`: element i of the output lives at
//   soa_out_base + i * 4
//
// For every record i in [0, record_count), in increasing order: touch()
// the SOURCE field address, then touch() the DESTINATION address --
// EXACTLY ONCE each. A genuinely elementwise transform touches exactly
// 2 * record_count addresses total (one read + one write per element) --
// touching another field, re-touching a record's other bytes, or
// touching any address twice, is not elementwise and inflates that count.
void aos_field_to_soa(long aos_base, long soa_out_base, int record_count,
                       int field_count, int field_index);
