#include "sol.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

ByteBuffer::ByteBuffer(const unsigned char* bytes, int n, bool is_poisoned)
    : data(nullptr), size(n), poisoned(is_poisoned) {
    if (n > 0) {
        data = (unsigned char*)std::malloc((size_t)n);
        std::memcpy(data, bytes, (size_t)n);
    }
}

ByteBuffer::ByteBuffer(const ByteBuffer& other)
    : data(nullptr), size(0), poisoned(false) {
    if (other.poisoned) {
        throw std::runtime_error("copy failed");
    }
    size = other.size;
    poisoned = other.poisoned;
    if (size > 0) {
        data = (unsigned char*)std::malloc((size_t)size);
        std::memcpy(data, other.data, (size_t)size);
    }
}

ByteBuffer::~ByteBuffer() {
    std::free(data);
}

void ByteBuffer::swap_with(ByteBuffer& other) noexcept {
    unsigned char* td = data;
    int ts = size;
    bool tp = poisoned;
    data = other.data;
    size = other.size;
    poisoned = other.poisoned;
    other.data = td;
    other.size = ts;
    other.poisoned = tp;
}

ByteBuffer& ByteBuffer::operator=(ByteBuffer rhs) {
    swap_with(rhs);
    return *this;
    // rhs (now holding *this's OLD contents) is destroyed here, freeing them.
}
