#pragma once

// Harness apparatus (declared here, DEFINED in main.cpp): a deterministic
// 4-way, 32-set, 64-byte-line LRU cache (8192 bytes total).
//   reset_cache()      -- clear it.
//   touch(addr)        -- an ordinary (temporal) load/store: goes
//                          through the cache normally (allocates a line
//                          on a miss, may evict something else).
//   touch_nt(addr)      -- a non-temporal (streaming) store: writes
//                          straight to memory. It does NOT consult or
//                          allocate into the cache at all -- it can
//                          never itself cause a miss, and it never
//                          evicts anything already resident.
//   miss_count()        -- running total misses recorded by touch()
//                          (touch_nt never contributes).
void reset_cache();
void touch(long byte_addr);
void touch_nt(long byte_addr);
long miss_count();

// Harness experiment (declared here, DEFINED in main.cpp): resets the
// cache, warms a small fixed 2048-byte "hot" region H with ordinary
// touch() accesses (simulating data another part of the program still
// needs), then writes `working_set_bytes` of a separate buffer -- via
// touch() if use_nt is false, via touch_nt() if use_nt is true. It then
// re-reads H (did writing the buffer evict it?) and, only if
// reused_soon is true, also re-reads the buffer itself (was the data
// you just wrote still resident for the reread?). Returns the TOTAL
// miss count accrued by both re-reads combined -- the real cost of
// whichever store strategy was used for this scenario.
long run_workload(long working_set_bytes, bool use_nt, bool reused_soon);

// Decide whether non-temporal stores beat ordinary temporal stores for
// writing `working_set_bytes` bytes in this scenario. Call
// run_workload() once with use_nt=false and once with use_nt=true
// (same working_set_bytes, same reused_soon both times) and return true
// iff the non-temporal run's total cost is STRICTLY LOWER.
bool nt_stores_help(long working_set_bytes, bool reused_soon);
