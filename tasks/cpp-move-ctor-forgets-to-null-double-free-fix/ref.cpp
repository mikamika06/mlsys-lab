#include "sol.hpp"

Buffer::Buffer(int value) : id_(heap::alloc(value)) {}

Buffer::~Buffer() { heap::release(id_); }

Buffer::Buffer(const Buffer& other) : id_(heap::alloc(heap::read(other.id_))) {}

Buffer::Buffer(Buffer&& other) noexcept : id_(other.id_) {
    other.id_ = 0;                       // the whole point: the source owns nothing now
}

Buffer& Buffer::operator=(const Buffer& other) {
    if (this == &other) return *this;    // self-assignment must not free our own id
    heap::release(id_);
    id_ = heap::alloc(heap::read(other.id_));
    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;
    heap::release(id_);                  // drop what we already owned
    id_ = other.id_;
    other.id_ = 0;
    return *this;
}

int Buffer::value() const { return heap::read(id_); }
