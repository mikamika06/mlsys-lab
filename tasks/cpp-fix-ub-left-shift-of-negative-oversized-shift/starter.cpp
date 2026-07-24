#include "sol.hpp"

// BUG: left-shifting a negative signed int, or shifting by >= the type's
// bit-width, is undefined behavior in C++ -- the compiler is free to do
// anything, including producing a result that does not match the
// well-defined two's-complement answer.
void process_shifts(const int* values, const int* shift_amounts, long* results, int n) {
    for (int i = 0; i < n; i++) {
        results[i] = (long)(values[i] << shift_amounts[i]);
    }
}
