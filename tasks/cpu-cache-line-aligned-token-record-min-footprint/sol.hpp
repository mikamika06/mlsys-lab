#pragma once
#include <cstddef>

// ============================================================================
// Deterministic fully-associative cache model (FIXED — defined in main.cpp).
// 64 lines x 64 bytes/line = 4096-byte cache, true LRU eviction. touch_byte(a)
// simulates one memory access to byte address `a`: it maps to line a/64, and
// counts a MISS whenever that line was not already resident (bringing it in
// evicts the least-recently-used line if the cache is full).
// ============================================================================
void reset_cache();
void touch_byte(long addr);
long miss_count();

// ============================================================================
// A "token" record needs 5 logical fields:
//   id     uint32_t   HOT  — read on every pass over the token array
//   count  uint32_t   HOT  — read AND incremented on every pass
//   flags  uint8_t    HOT  — read and toggled on every pass
//   name   char[24]   COLD — part of the record, never touched by the hot pass
//   ts     uint64_t   COLD — part of the record, never touched by the hot pass
//
// Design ONE record type (a real struct — sizeof/offsetof come straight from
// the compiler) that keeps the 3 HOT fields adjacent and unpadded at the
// front so a single 64-byte cache line covers all of a record's hot data,
// while placing the two COLD fields after them with the least alignment
// padding possible, so sizeof(record) is as small as it can be. The driver
// only ever touches id/count/flags in its measured loop; name/ts exist
// purely to inflate the record the way a real token struct would, so their
// order and padding cost is exactly what you are optimizing away.
//
// Report your layout through these five functions (implemented with
// sizeof/offsetof against your own struct, defined in your .cpp file):
size_t record_size();   // sizeof(your record)
size_t offset_id();     // offsetof(your record, id)
size_t offset_count();  // offsetof(your record, count)
size_t offset_flags();  // offsetof(your record, flags)
size_t offset_name();   // offsetof(your record, name)
size_t offset_ts();     // offsetof(your record, ts)
