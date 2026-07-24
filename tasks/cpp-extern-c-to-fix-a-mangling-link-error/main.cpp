// FIXED driver. Plays both sides of the link: the C++ implementation
// (`cpp_add`) and the C-style consumer, which can only ever call a plain,
// unmangled symbol named `add` -- exactly what a real `.c` file (or any
// non-C++ caller) requires. `add`'s definition lives in ref.cpp/solve.cpp;
// it either genuinely link here, or it genuinely doesn't.
#include <cstdio>

#include "sol.hpp"

int cpp_add(int a, int b) { return a + b; }

// The C-style consumer's expectation: it wants the plain C symbol `add`,
// not whatever the C++ compiler would mangle a same-named C++ function to
// (e.g. `_Z3addii` under the Itanium ABI). This is what actually forces
// solve.cpp/ref.cpp to export `add` with C linkage to link at all.
extern "C" int add(int a, int b);

int main() {
    printf("%d\n", add(19, 23));
    return 0;
}
