#pragma once

// Instrumented allocator, DEFINED in main.cpp. Every heap buffer a Capsule
// owns must flow through these two so the driver can count allocations
// against frees and catch a leak (alloc > free) or a double-free (free >
// alloc, and a real crash on this real allocator).
unsigned char* arena_alloc(int n);
void arena_free(unsigned char* p);

// A capsule owning a heap buffer allocated with arena_alloc -- the C++ side
// of what a pybind11 function does when it returns a numpy array backed by
// C++-allocated memory tied to a capsule base object: ownership is UNIQUE,
// it moves like std::unique_ptr, and the buffer must survive being returned
// by value, stored in a container, and reshuffled by moves -- with the
// underlying memory freed EXACTLY ONCE, whenever the capsule that still
// owns it is destroyed.
struct Capsule {
    unsigned char* data;
    int size;

    Capsule() : data(nullptr), size(0) {}
    Capsule(const Capsule&) = delete;
    Capsule& operator=(const Capsule&) = delete;

    // Move ctor: adopt other's buffer, leave other empty (data == nullptr)
    // so other's destructor becomes a no-op.
    Capsule(Capsule&& other) noexcept;

    // Move assign: release whatever *this currently owns (arena_free), then
    // adopt other's buffer and leave other empty.
    Capsule& operator=(Capsule&& other) noexcept;

    // Release the owned buffer (arena_free) if this capsule still owns one.
    ~Capsule();
};

// Factory: allocate `n` bytes with arena_alloc, fill byte i with
// (unsigned char)(i * mult), and return a Capsule owning that buffer.
Capsule make_capsule(int n, int mult);
