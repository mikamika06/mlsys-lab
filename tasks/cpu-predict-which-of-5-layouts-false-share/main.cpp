#include <cstdio>

#include "sol.hpp"

// FIXED driver: classify the 5 layouts under a real 64-byte cache line.
int main() {
    std::array<bool, 5> result = classify_layouts(64);
    for (bool b : result) {
        printf("%d\n", b ? 1 : 0);
    }
    return 0;
}
