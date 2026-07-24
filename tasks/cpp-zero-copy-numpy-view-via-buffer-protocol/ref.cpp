#include "sol.hpp"

// Correct: alias arr's own memory, no copy.
ArrayView make_zero_copy_view(double* arr, long n) {
    ArrayView v;
    v.buf = arr;
    v.len = n;
    v.itemsize = (long)sizeof(double);
    v.stride = (long)sizeof(double);
    return v;
}
