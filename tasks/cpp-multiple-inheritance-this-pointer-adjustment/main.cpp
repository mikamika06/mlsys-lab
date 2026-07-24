#include <cstdio>
#include "sol.hpp"

// FIXED driver. Prints the this-pointer adjustment for each base and
// sizeof(Derived).
int main() {
    std::size_t offs[4] = {0, 0, 0, 0};
    base_offsets(offs);
    const char* names[4] = {"B1", "B2", "B3", "sizeof"};
    for (int i = 0; i < 4; i++) printf("%s=%zu\n", names[i], offs[i]);
    return 0;
}
