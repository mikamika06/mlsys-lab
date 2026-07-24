#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

// Bounded single-producer / single-consumer (SPSC) ring buffer of int32 values.
//
// A real SPSC queue publishes one producer index and one consumer index with
// release/acquire ordering so the two threads never lose or duplicate an
// element. Here you implement the *index arithmetic* that guarantees those two
// invariants; the driver exercises it single-threaded and deterministically.
//
// Storage is provided: `buf_` has exactly `cap_` slots, and `head_` / `tail_`
// are the two indices you maintain. You choose their exact meaning, but the
// observable behaviour must be:
//
//   try_push(v)  : if the buffer is full, return false and change nothing;
//                  otherwise store v, advance the producer side, return true.
//   try_pop(out) : if the buffer is empty, return false and change nothing;
//                  otherwise copy the OLDEST buffered element into *out,
//                  advance the consumer side, return true.
//   size()       : number of elements currently buffered (0 .. capacity()).
//   capacity()   : maximum number of elements.
//
// FIFO order must be preserved and every accepted element must be popped
// exactly once: no loss, no duplication, no reordering.
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buf_(capacity), cap_(capacity), head_(0), tail_(0) {}

    bool   try_push(int32_t v);      // learner implements
    bool   try_pop(int32_t* out);    // learner implements
    size_t size() const;             // learner implements

    size_t capacity() const { return cap_; }

private:
    std::vector<int32_t> buf_;  // backing storage, exactly cap_ slots
    size_t cap_;                // capacity (number of usable slots)
    size_t head_;               // consumer-side index (you define the meaning)
    size_t tail_;               // producer-side index (you define the meaning)
};
