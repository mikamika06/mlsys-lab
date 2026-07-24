#include "sol.hpp"

// BUG: this looks correct and compiles fine on its own -- but without
// `extern "C"`, the C++ compiler mangles `add`'s exported symbol name (it
// encodes the parameter types into the name so overloading works). The
// C-style consumer in main.cpp asks the linker for the plain symbol `add`
// and won't find it. Fix the language linkage.
int add(int a, int b) {
    return cpp_add(a, b);
}
