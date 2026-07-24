#include <cstdio>
#include <string>
#include "sol.hpp"

int main() {
    const char* names[] = {
        "_Z1fv", "_Z1fi", "_Z3fooii", "_Z4funcPd", "_Z4funcPKd", "_Z4funcRKd",
        "_ZN1S1fEv", "_ZN1S1fEi", "_ZN5Outer5Inner4funcEv", "_ZNK1S1fEv",
        "_ZNK5Outer1fEPi", "_ZN4Math3addEii",
    };
    for (const char* n : names) {
        std::string d = demangleOne(n);
        printf("%s -> %s\n", n, d.c_str());
    }
    return 0;
}
