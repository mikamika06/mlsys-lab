#include "sol.hpp"

// Row-major matrix, row-major traversal: consecutive touches (col
// incrementing) stay within the same 64-byte cache line for 16 columns at
// a time before moving to the next line -- maximal spatial locality.
void access_pattern(int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            long addr = (long)(row * N + col) * 4;
            touch(addr);
        }
    }
}
