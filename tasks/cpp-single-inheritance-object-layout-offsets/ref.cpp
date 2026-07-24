#include "sol.hpp"
#include <cstddef>

// Reference: read the real offsets straight from the compiler (the Itanium ABI
// oracle). offsetof gives the member offsets; the vptr sits at offset 0 for the
// primary base under single inheritance.
void derived_layout(std::size_t offs[8]) {
    offs[0] = 0;                     // vptr: object base address, offset 0
    offs[1] = offsetof(Derived, a);
    offs[2] = offsetof(Derived, b);
    offs[3] = offsetof(Derived, c);
    offs[4] = offsetof(Derived, d);
    offs[5] = offsetof(Derived, e);
    offs[6] = offsetof(Derived, f);
    offs[7] = sizeof(Derived);
}
