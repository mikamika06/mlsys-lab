#include "sol.hpp"

// BROKEN: this LOOKS safe because `value` is std::atomic<long>, but the
// increment is done as two separate atomic operations -- a load(), then
// (after a plain, non-atomic +1 in a local) a store() -- instead of one
// atomic read-modify-write. Two threads can both load() the same old value
// before either has stored, both compute old+1, and both store() the same
// result: one increment vanishes. Fix it with a single RMW op such as
// counter.value.fetch_add(1, ...).
void increment(SharedCounter& counter) {
    long v = counter.value.load(std::memory_order_relaxed);
    v = v + 1;
    counter.value.store(v, std::memory_order_relaxed);
}
