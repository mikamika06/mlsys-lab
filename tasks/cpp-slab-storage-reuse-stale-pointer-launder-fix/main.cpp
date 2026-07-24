#include <cstdio>
#include "sol.hpp"

// FIXED driver.
int main() {
    int result = slab_reuse_demo();
    printf("%d\n", result);
    return 0;
}
