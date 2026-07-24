#include <cstdio>
#include "sol.hpp"

// FIXED driver. Do not edit. 12 fixed signatures covering void, int,
// double, char, long, float, bool, pointers to these, and the
// empty-parameter-list case -- calls the learner's mangler and prints
// every result.
int main() {
    std::vector<std::string> sigs = {
        "void foo()",
        "int bar(int)",
        "double baz(double, double)",
        "char qux(int, double)",
        "void quux(int*)",
        "int* corge(char, double*)",
        "void grault(long)",
        "double garply(bool)",
        "float waldo(float)",
        "long fred(int*, char*)",
        "void* plugh(int)",
        "char xyzzy(void)",
    };

    auto out = mangle_signatures(sigs);
    for (const auto& s : out) printf("%s\n", s.c_str());
    return 0;
}
