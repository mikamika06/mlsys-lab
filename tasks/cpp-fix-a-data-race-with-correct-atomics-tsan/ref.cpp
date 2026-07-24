#include "sol.hpp"

void increment(SharedCounter& counter) {
    // A single atomic read-modify-write instruction: no other thread's
    // fetch_add can interleave inside it, so no increment is ever lost.
    // (The memory_order argument affects ordering relative to OTHER memory
    // operations, not whether THIS increment itself is atomic -- relaxed is
    // sufficient for a plain counter with no data being published by it.)
    counter.value.fetch_add(1, std::memory_order_relaxed);
}
