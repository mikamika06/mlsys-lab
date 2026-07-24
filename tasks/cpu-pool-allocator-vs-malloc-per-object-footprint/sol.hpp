#pragma once

// Pinned allocator model constants (defined in main.cpp): every malloc'd
// block carries a fixed HEADER_BYTES of bookkeeping overhead (size,
// free-list links, etc.) in front of the usable payload, and the total
// block size (header + payload) is always rounded UP to a multiple of
// ALIGN_BYTES.
extern const int HEADER_BYTES;
extern const int ALIGN_BYTES;

// LEARNER IMPLEMENTS.
//
// Total bytes actually reserved to hold `count` separate objects of
// `obj_bytes` each, allocated with ONE malloc() call per object: every
// individual allocation pays its own HEADER_BYTES of overhead, and each
// individual (header + payload) size is rounded up to ALIGN_BYTES
// *before* being multiplied by count.
long malloc_per_object_footprint(int count, int obj_bytes);

// Total bytes reserved by a POOL allocator: a SINGLE allocation holding
// all `count` objects back-to-back with no per-object header -- only
// ONE HEADER_BYTES is paid for the whole pool, and the pool's total
// payload (count * obj_bytes) is rounded up to ALIGN_BYTES once, after
// adding that one header.
long pool_footprint(int count, int obj_bytes);

// footprint_ratio = malloc_per_object_footprint(count, obj_bytes)
//                    / (double) pool_footprint(count, obj_bytes)
// How many times bigger the per-object malloc footprint is than the
// pool's, for the same count and object size.
double footprint_ratio(int count, int obj_bytes);
