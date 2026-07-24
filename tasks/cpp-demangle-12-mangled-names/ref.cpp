#include "sol.hpp"
#include <cxxabi.h>
#include <cstdlib>

// The reference forwards to the REAL Itanium demangler shipped in
// libc++abi -- the exact same code path the linker/debugger use -- so the
// expected output is never hand-typed anywhere in this task.
std::string demangleOne(const std::string& mangled) {
    int status = 0;
    char* out = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
    std::string result = (status == 0 && out) ? std::string(out) : std::string("<error>");
    free(out);
    return result;
}
