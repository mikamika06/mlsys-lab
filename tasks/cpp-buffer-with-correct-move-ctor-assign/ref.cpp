#include "sol.hpp"

Buffer::Buffer(long size_) : ptr(tracked_alloc(size_)), size(size_) {}

Buffer::~Buffer() {
    if (ptr != 0) tracked_free(ptr);
    ptr = 0;
    size = 0;
}

Buffer::Buffer(const Buffer& other) : ptr(tracked_alloc(other.size)), size(other.size) {
    tracked_deep_copy(ptr, other.ptr, other.size);
}

Buffer::Buffer(Buffer&& other) noexcept : ptr(other.ptr), size(other.size) {
    other.ptr = 0;
    other.size = 0;
}

Buffer& Buffer::operator=(const Buffer& other) {
    if (&other == this) return *this;
    if (ptr != 0) tracked_free(ptr);
    ptr = tracked_alloc(other.size);
    size = other.size;
    tracked_deep_copy(ptr, other.ptr, other.size);
    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (&other == this) return *this;
    if (ptr != 0) tracked_free(ptr);
    ptr = other.ptr;
    size = other.size;
    other.ptr = 0;
    other.size = 0;
    return *this;
}
