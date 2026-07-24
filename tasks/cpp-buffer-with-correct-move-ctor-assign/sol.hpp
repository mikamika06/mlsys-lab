#pragma once

// ---------------------------------------------------------------------------
// Harness-side instrumented allocator (DEFINED in main.cpp).
//
// A tiny fake heap: tracked_alloc() hands back a nonzero integer "id" (never
// a real pointer, so nothing actually corrupts memory) and bumps an alloc
// counter. tracked_free(id) is a no-op when id == 0, otherwise it retires the
// id and bumps a free counter. tracked_deep_copy() just bumps a deep-copy
// counter. Read the running totals with the stats_* accessors.
// ---------------------------------------------------------------------------
long tracked_alloc(long size);
void tracked_free(long id);
void tracked_deep_copy(long dst_id, long src_id, long size);

long stats_allocs();
long stats_frees();
long stats_deep_copies();

// ---------------------------------------------------------------------------
// Buffer: a pointer-owning resource handle. Implement the Rule of Five so
// that copies deep_copy and moves NEVER deep_copy (they just steal the
// source's id and null the source out).
//
//   Buffer(size)            construct: tracked_alloc(size); size field = size.
//   ~Buffer()                destroy: tracked_free(ptr) if ptr != 0.
//   Buffer(const Buffer&)    copy ctor: tracked_alloc(other.size), then
//                            tracked_deep_copy(new_id, other.ptr, other.size).
//   Buffer(Buffer&&)         move ctor: steal other.ptr/other.size, then set
//                            other.ptr = 0, other.size = 0. No alloc, no
//                            deep_copy, no free.
//   operator=(const Buffer&) copy assign: if (&other == this) do nothing and
//                            return *this. Otherwise tracked_free(ptr) if
//                            nonzero, tracked_alloc(other.size), then
//                            tracked_deep_copy(new_id, other.ptr, other.size).
//   operator=(Buffer&&)      move assign: if (&other == this) do nothing and
//                            return *this. Otherwise tracked_free(ptr) if
//                            nonzero, steal other.ptr/other.size, then set
//                            other.ptr = 0, other.size = 0.
//
// `ptr` is the id returned by tracked_alloc (0 means "owns nothing").
// ---------------------------------------------------------------------------
class Buffer {
public:
    long ptr = 0;
    long size = 0;

    Buffer() = default;
    explicit Buffer(long size_);
    ~Buffer();

    Buffer(const Buffer& other);
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(const Buffer& other);
    Buffer& operator=(Buffer&& other) noexcept;
};
