#include "sol.hpp"

long contiguous_load(long base, int n, int vec_width, int elem_bytes) {
    long chunks = n / vec_width;
    for (long c = 0; c < chunks; ++c) {
        touch(base + c * (long)vec_width * elem_bytes);
    }
    return chunks;
}

long gather_load(long base, const int* idx, int n, int elem_bytes) {
    for (int k = 0; k < n; ++k) {
        touch(base + (long)idx[k] * elem_bytes);
    }
    return n;
}
