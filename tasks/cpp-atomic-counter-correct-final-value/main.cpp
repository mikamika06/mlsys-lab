#include <cstdio>
#include "sol.hpp"

// FIXED driver: a handful of (num_threads, increments_per_thread) cases.
// Every case must print exactly num_threads * increments_per_thread.
int main() {
    struct Case { int t; int i; };
    const Case cases[] = { {1, 500}, {4, 1000}, {8, 125}, {6, 2000} };

    for (const auto& c : cases) {
        long v = atomic_counter_final_value(c.t, c.i);
        printf("%ld\n", v);
    }
    return 0;
}
