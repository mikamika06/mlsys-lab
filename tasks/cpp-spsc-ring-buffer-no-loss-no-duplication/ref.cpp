#include "sol.hpp"

// Reference: head_ and tail_ are monotonically increasing counters (total pops
// and total pushes). The physical slot is index % cap_. size = tail_ - head_,
// which makes the full/empty distinction unambiguous.

bool RingBuffer::try_push(int32_t v) {
    if (tail_ - head_ >= cap_) return false;   // full
    buf_[tail_ % cap_] = v;
    ++tail_;
    return true;
}

bool RingBuffer::try_pop(int32_t* out) {
    if (tail_ == head_) return false;          // empty
    *out = buf_[head_ % cap_];
    ++head_;
    return true;
}

size_t RingBuffer::size() const {
    return tail_ - head_;
}
