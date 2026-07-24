#pragma once

// ============================================================================
// Two store mechanisms into the SAME modeled 64-byte-line cache (both
// FIXED — do not modify; defined in main.cpp). Both genuinely write
// `*p = v` to real memory; only whether the destination's cache line gets
// registered as resident differs, exactly like a normal store (allocates a
// line, evicting something else if the cache is full) versus a real
// non-temporal / streaming store (bypasses the cache entirely -- e.g. x86
// MOVNTPS/MOVNTDQ, or the write-combining path a streaming ARM store
// reaches -- the write lands in memory through a write-combining buffer
// and never occupies a cache line).
//
// store_temporal(p, v):    *p = v, AND registers the 64-byte line
//                           containing p as touched.
// store_nontemporal(p, v): *p = v, WITHOUT touching any cache line.
// dst_lines_touched():     distinct lines registered as touched since the
//                           last reset_dst_lines().
// ============================================================================
void reset_dst_lines();
void store_temporal(float* p, float v);
void store_nontemporal(float* p, float v);
int dst_lines_touched();

// ============================================================================
// Copy src[0..n) into dst[0..n), value for value. `dst` is a large output
// buffer written exactly once that will not be re-read again any time
// soon (the classic streaming-store use case -- e.g. flushing a large
// computed result out to memory): write every element of dst through
// store_nontemporal, never store_temporal, so streaming the copy out
// doesn't evict anything useful that's already resident in cache.
// ============================================================================
void stream_copy(const float* src, float* dst, int n);
