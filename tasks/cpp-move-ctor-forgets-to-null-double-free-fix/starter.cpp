#include "sol.hpp"

// The move constructor below is the bug: it steals the id but leaves the source
// still holding it, so both destructors release the same allocation.
// Fix it, and finish the other special members so Buffer is a correct RAII type.

Buffer::Buffer(int value) : id_(heap::alloc(value)) {}

Buffer::~Buffer() { heap::release(id_); }

Buffer::Buffer(const Buffer& other) : id_(heap::alloc(heap::read(other.id_))) {}

Buffer::Buffer(Buffer&& other) noexcept : id_(other.id_) {
    // BUG: other.id_ is left untouched
}

Buffer& Buffer::operator=(const Buffer& other) {
    heap::release(id_);
    id_ = heap::alloc(heap::read(other.id_));
    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    id_ = other.id_;
    return *this;
}

int Buffer::value() const { return heap::read(id_); }
