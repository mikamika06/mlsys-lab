#include "sol.hpp"

// Reference: read the real adjustments straight from the compiler (the
// Itanium ABI oracle) -- static_cast across a base performs exactly the
// this-pointer adjustment being asked for; the pointer difference IS the
// answer.
void base_offsets(std::size_t offs[4]) {
    Derived d{};
    Derived* dp = &d;
    offs[0] = (std::size_t)((char*)static_cast<B1*>(dp) - (char*)dp);
    offs[1] = (std::size_t)((char*)static_cast<B2*>(dp) - (char*)dp);
    offs[2] = (std::size_t)((char*)static_cast<B3*>(dp) - (char*)dp);
    offs[3] = sizeof(Derived);
}
