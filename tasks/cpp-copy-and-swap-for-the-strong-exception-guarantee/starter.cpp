#include "sol.hpp"

// TODO: implement each piece of the copy-and-swap idiom described in
// sol.hpp / task.md.

ByteBuffer::ByteBuffer(const unsigned char* bytes, int n, bool is_poisoned)
    : data(nullptr), size(0), poisoned(is_poisoned) {
    (void)bytes; (void)n;
    // your code here
}

ByteBuffer::ByteBuffer(const ByteBuffer& other)
    : data(nullptr), size(0), poisoned(false) {
    (void)other;
    // your code here
}

ByteBuffer::~ByteBuffer() {
    // your code here
}

void ByteBuffer::swap_with(ByteBuffer& other) noexcept {
    (void)other;
    // your code here
}

ByteBuffer& ByteBuffer::operator=(ByteBuffer rhs) {
    (void)rhs;
    // your code here
    return *this;
}
