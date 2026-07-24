#include "sol.hpp"

Capsule::Capsule(Capsule&& other) noexcept : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
}

Capsule& Capsule::operator=(Capsule&& other) noexcept {
    if (this != &other) {
        arena_free(data);       // release whatever *this currently owns
        data = other.data;
        size = other.size;
        other.data = nullptr;   // other no longer owns it
        other.size = 0;
    }
    return *this;
}

Capsule::~Capsule() {
    arena_free(data);           // no-op (via the p==nullptr guard) if empty
}

Capsule make_capsule(int n, int mult) {
    Capsule c;
    c.data = arena_alloc(n);
    c.size = n;
    for (int i = 0; i < n; i++) c.data[i] = (unsigned char)(i * mult);
    return c;
}
