#include <cstdio>
#include "sol.hpp"

// FIXED driver: a mix of negative values, positive values, in-range shifts,
// and out-of-range (>= 32) shifts.
int main() {
    const int N = 8;
    const int values[N]         = {-1,  -1, -100, 7,  1,          -2147483647 - 1, 5,  -16};
    const int shift_amounts[N]  = {31,  35,   3,  40, 0,          1,               33, 32};

    long results[N] = {};
    process_shifts(values, shift_amounts, results, N);

    for (int i = 0; i < N; i++) {
        printf("%ld\n", results[i]);
    }
    return 0;
}
