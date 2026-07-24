#include "sol.hpp"

// TODO: implement the SPSC ring buffer so the consumer receives exactly the
// produced sequence -- no loss, no duplication, FIFO order preserved.
// Use head_ / tail_ (and buf_ / cap_) to track the producer and consumer sides.
// The stubs below compile but are wrong: nothing is ever stored or returned.

bool RingBuffer::try_push(int32_t v) {
    (void)v;
    return false;  // TODO: store v and advance the producer side
}

bool RingBuffer::try_pop(int32_t* out) {
    (void)out;
    return false;  // TODO: return the oldest element and advance the consumer side
}

size_t RingBuffer::size() const {
    return 0;      // TODO: number of buffered elements
}
