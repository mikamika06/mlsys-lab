#include "sol.hpp"
#include <cstddef>

// TODO: Derive the byte offsets of the vptr and every member inside a Derived
// object under the Itanium C++ ABI (LP64), then write them into offs[0..7]:
//   [0]=vptr [1]=a [2]=b [3]=c [4]=d [5]=e [6]=f [7]=sizeof(Derived).
//
// Reason it out:
//   - The vptr comes first, at offset 0 (single inheritance reuses the base vptr).
//   - Members follow in declaration order, each rounded up to its own alignment,
//     inserting padding as needed.
//   - Derived's own members may be placed in the *tail padding* of the Base
//     subobject (Itanium ABI reuses the base's data-size, not its full sizeof).
//   - sizeof(Derived) is the final data size rounded up to the class alignment.
void derived_layout(std::size_t offs[8]) {
    // your code here
}
