#include <cstdio>
#include "sol.hpp"

// FIXED driver: simulates n_pushes push_back() calls for three
// (n_pushes, growth_factor) scenarios, reallocating via grow_capacity()
// whenever size == capacity, and prints the capacity AFTER each push.

static void run_case(int n_pushes, double growth_factor) {
    int size = 0, cap = 0;
    for (int i = 0; i < n_pushes; ++i) {
        if (size == cap) {
            cap = grow_capacity(cap, growth_factor);
        }
        ++size;
        printf("%d ", cap);
    }
    printf("\n");
}

int main() {
    run_case(10, 2.0);
    run_case(20, 1.5);
    run_case(30, 1.25);
    printf("sizeof(VectorHeader)=%d\n", static_cast<int>(sizeof(VectorHeader)));
    return 0;
}
