#include <cstdio>
#include "sol.hpp"

// FIXED driver translation unit. This global's OWN dynamic initializer
// calls get_b_value() DURING this TU's static initialization phase — and
// because main.cpp is listed FIRST on the compile command
// (`clang++ ... main.cpp <src>`), this TU's dynamic initializers run
// before the other TU's. If get_b_value() is backed by a plain
// namespace-scope global in the OTHER translation unit, this call happens
// before that global has been constructed.
int g_a_derived = get_b_value() + 100;

int main() {
    printf("g_a_derived=%d\n", g_a_derived);
    return 0;
}
