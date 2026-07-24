#pragma once

// An owning heap buffer of bytes that assigns via the COPY-AND-SWAP idiom,
// which gives the STRONG exception guarantee: if copying the source throws,
// *this must be left byte-identical to what it was before the assignment.
//
// A ByteBuffer built with is_poisoned=true THROWS std::runtime_error when
// copy-constructed (models a copy that can fail, e.g. an allocation
// failure) -- this is how the driver forces operator= to fail partway
// through.
struct ByteBuffer {
    unsigned char* data;
    int size;
    bool poisoned;

    ByteBuffer() : data(nullptr), size(0), poisoned(false) {}

    // Owns a deep copy of bytes[0..n).
    ByteBuffer(const unsigned char* bytes, int n, bool is_poisoned);

    // Deep copy. Must THROW std::runtime_error (and allocate/change
    // nothing) if other.poisoned is true; otherwise copy other's bytes.
    ByteBuffer(const ByteBuffer& other);

    ~ByteBuffer();

    // Exchange contents with `other`. Must not allocate and must not throw.
    void swap_with(ByteBuffer& other) noexcept;

    // Copy-and-swap assignment: takes its argument BY VALUE, so the copy of
    // the source (and any throw from it) happens while constructing `rhs`,
    // BEFORE this body ever runs. If that copy throws, *this is untouched
    // because control never reaches here. Otherwise, swap rhs into *this.
    ByteBuffer& operator=(ByteBuffer rhs);
};
