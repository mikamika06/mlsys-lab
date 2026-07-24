#pragma once
#include <atomic>

// A counter that must be safe to increment from many threads concurrently,
// with no lost updates.
struct SharedCounter {
    std::atomic<long> value{0};
};

// Increment `counter` by exactly 1, safely under concurrent callers.
//
// Must perform the increment as a SINGLE ATOMIC read-modify-write (e.g.
// counter.value.fetch_add(1, ...)). Separately load()-ing the current value,
// adding 1 in a local, and store()-ing it back is a DATA RACE even though
// `value`'s TYPE is std::atomic<long>: two threads can both load() the same
// old value, both compute old+1, and both store() the same result -- one of
// the two increments is silently lost. Wrapping a variable in std::atomic
// only makes each INDIVIDUAL load/store atomic; it does not make a
// load-then-store sequence atomic as a whole.
void increment(SharedCounter& counter);
