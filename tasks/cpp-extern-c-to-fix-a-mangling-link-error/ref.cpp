#include "sol.hpp"

// Reference: give `add` C language linkage so it exports the plain,
// unmangled symbol `add` the C-style consumer in main.cpp links against.
extern "C" int add(int a, int b) {
    return cpp_add(a, b);
}
