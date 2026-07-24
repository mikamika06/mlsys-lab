// FIXED driver. Trivial on purpose: classify_type<T>() is a function
// template, so both its definition and every explicit instantiation used
// to test it have to live in whichever .cpp defines it (see sol.hpp).
#include "sol.hpp"

int main() {
    run_declaration_tests();
    return 0;
}
