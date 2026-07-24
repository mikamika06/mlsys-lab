#include "sol.hpp"

// TODO: implement the capsule's move ctor, move assign, destructor, and the
// make_capsule factory. See sol.hpp for exactly what each one must do.

Capsule::Capsule(Capsule&& other) noexcept : data(nullptr), size(0) {
    (void)other;
    // your code here
}

Capsule& Capsule::operator=(Capsule&& other) noexcept {
    (void)other;
    // your code here
    return *this;
}

Capsule::~Capsule() {
    // your code here
}

Capsule make_capsule(int n, int mult) {
    (void)n; (void)mult;
    // your code here
    return Capsule();
}
