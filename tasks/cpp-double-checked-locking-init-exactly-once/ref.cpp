#include "sol.hpp"

bool fast_check(const SingletonState& s) {
    return s.ready.load(std::memory_order_acquire);
}

bool try_init(SingletonState& s) {
    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.ready.load(std::memory_order_relaxed)) {
        return false; // someone else already finished while we were waiting
    }
    s.init_count++;
    s.ready.store(true, std::memory_order_release);
    return true;
}
