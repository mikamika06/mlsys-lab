#pragma once
#include <atomic>
#include <mutex>

struct SingletonState {
    std::atomic<bool> ready{false};
    std::mutex mtx;
    int init_count = 0; // must end up exactly 1, no matter how callers race
};

// Fast path: a lock-free peek at whether the singleton is already
// initialized. Must not take the lock and must not modify `s`.
bool fast_check(const SingletonState& s);

// Slow path: called only after a fast_check() that returned false. Must
// take s.mtx, RE-CHECK s.ready now that the lock is held (the "double" in
// double-checked locking — another caller may have finished initializing
// while this caller was on its way to the lock), initialize (increment
// s.init_count, set s.ready) only if still not ready, then release the
// lock. Returns true iff THIS call performed the initialization.
bool try_init(SingletonState& s);
