#include <cstdio>
#include "sol.hpp"

int main() {
    // "Native" layout: an 8x16 row-major array of 4-byte floats
    // (row_stride=16=C, col_stride=1) -- each row is exactly one
    // 64-byte line (16 * 4 = 64).
    const long NATIVE_RS = 16, NATIVE_CS = 1;
    const int NATIVE_R = 8, NATIVE_C = 16, NATIVE_EB = 4, LINE = 64;

    long addr_native = element_addr(0, 3, 5, NATIVE_RS, NATIVE_CS, NATIVE_EB);
    long fetch_native_rowmajor =
        traversal_fetch_count(0, NATIVE_R, NATIVE_C, NATIVE_RS, NATIVE_CS, NATIVE_EB, LINE, true);
    long fetch_native_colmajor =
        traversal_fetch_count(0, NATIVE_R, NATIVE_C, NATIVE_RS, NATIVE_CS, NATIVE_EB, LINE, false);

    // "Transposed" VIEW of an 8x8 array of 8-byte doubles
    // (row_stride=1, col_stride=8) -- the same formula, just with the
    // stride-1 axis swapped to the row.
    const long TRANS_RS = 1, TRANS_CS = 8;
    const int TRANS_R = 8, TRANS_C = 8, TRANS_EB = 8;

    long addr_transposed = element_addr(0, 3, 5, TRANS_RS, TRANS_CS, TRANS_EB);
    long fetch_transposed_rowmajor =
        traversal_fetch_count(0, TRANS_R, TRANS_C, TRANS_RS, TRANS_CS, TRANS_EB, LINE, true);
    long fetch_transposed_colmajor =
        traversal_fetch_count(0, TRANS_R, TRANS_C, TRANS_RS, TRANS_CS, TRANS_EB, LINE, false);

    printf("addr_native=%ld addr_transposed=%ld "
           "fetch_native_rowmajor=%ld fetch_native_colmajor=%ld "
           "fetch_transposed_rowmajor=%ld fetch_transposed_colmajor=%ld\n",
           addr_native, addr_transposed,
           fetch_native_rowmajor, fetch_native_colmajor,
           fetch_transposed_rowmajor, fetch_transposed_colmajor);
    return 0;
}
