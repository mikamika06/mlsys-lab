#include <cstdio>
#include <cstddef>
#include "sol.hpp"

// FIXED driver. Do not edit. Calls the learner's derived_layout() and prints
// the offset of the vptr, every member, and sizeof(Derived), one per line.
int main() {
    std::size_t offs[8] = {0};
    derived_layout(offs);
    const char* names[8] = {"vptr", "a", "b", "c", "d", "e", "f", "sizeof"};
    for (int i = 0; i < 8; i++) printf("%s=%zu\n", names[i], offs[i]);
    return 0;
}
