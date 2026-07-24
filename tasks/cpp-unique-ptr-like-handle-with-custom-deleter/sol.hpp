#pragma once
#include <cstdint>

// ============================================================================
// Fake GPU driver  (FIXED — do not modify these three definitions).
// A single global counter is incremented by the custom deleter every time a
// live resource is freed. `inline` gives one shared instance across all
// translation units.
// ============================================================================

// Number of times a resource has been released (the "release counter").
inline int g_release_count = 0;

// The custom deleter. Frees one resource id. Releasing the empty id (0) is a
// no-op, so it never counts.
inline void gpu_release(std::uint64_t id) {
    if (id != 0) ++g_release_count;
}

// Acquire a fresh, non-zero, opaque resource id.
inline std::uint64_t gpu_acquire() {
    static std::uint64_t next = 1;
    return next++;
}

// ============================================================================
// Move-only RAII handle with a custom deleter  (LEARNER implements the members
// declared below, in solve.cpp — see task.md).
//
// Requirements:
//   * The destructor releases the owned id EXACTLY ONCE via gpu_release().
//   * The type is MOVE-ONLY (copy operations are deleted below).
//   * A moved-from handle owns nothing and releases nothing on destruction.
//   * Move-assignment first releases the currently-owned id, then steals the
//     source's id and leaves the source empty.
//   * reset() releases the owned id immediately and leaves the handle empty
//     (so a later destructor must not release it a second time).
// ============================================================================
class GpuHandle {
    std::uint64_t id_;
public:
    explicit GpuHandle(std::uint64_t id);              // take ownership of id
    ~GpuHandle();                                      // release owned id once

    GpuHandle(const GpuHandle&)            = delete;   // no copies
    GpuHandle& operator=(const GpuHandle&) = delete;   // no copies

    GpuHandle(GpuHandle&& other) noexcept;             // steal ownership
    GpuHandle& operator=(GpuHandle&& other) noexcept;  // release own, then steal

    std::uint64_t get() const;                         // owned id (0 if empty)
    void reset();                                      // release now, become empty
};
