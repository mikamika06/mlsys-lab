#pragma once

// Deterministic direct-mapped cache model (harness code, defined in
// main.cpp): 64-byte lines, 64 sets (4096 bytes total). touch_byte(addr)
// simulates reading the 4-byte value at byte address `addr` through this
// cache and counts a MISS whenever that line wasn't already resident.
void reset_cache();
void touch_byte(long addr);
long miss_count();

// n records of `stride` bytes each are laid out back to back starting at
// byte address `base`. Record i's "hot" 4-byte float field lives at byte
// address base + i*stride + hot_offset; its value (there is no real
// backing memory -- only the address space the cache model tracks) is
//   value(i) = (double)((i * 37) % 97) - 48.0
//
// Compute sum, min, and max of the hot field over all n records and
// write them into out[0], out[1], out[2]. `stride` is large (a fat
// struct with the hot field surrounded by cold ones), so each record's
// hot field lands in its own cache line -- every READ of it must go
// through touch_byte(address) exactly once per read. The bug this task
// is about is reading each record's hot field 3 separate times (one full
// scan for sum, one for min, one for max): with a working set far bigger
// than the cache, none of the 3 scans can reuse another's misses, so
// that costs 3x the true minimum. Fix it by computing all 3 reductions
// in a SINGLE pass over the records, touching each hot field once.
void hot_field_stats(long base, int stride, int hot_offset, int n, double* out);
