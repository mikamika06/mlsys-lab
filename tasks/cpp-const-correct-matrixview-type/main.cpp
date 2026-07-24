#include <cstdio>
#include "sol.hpp"

// FIXED driver. A 4x6 backing array is filled with a canary pattern, then a
// 3x3 view is opened over a sub-block of it (row_stride = 6 != cols = 3), so
// a naive `data[r * cols + c]` implementation that forgets the stride reads
// and writes the wrong cells. The view is written through as a plain
// (mutable) MatrixView, then read back through a `const MatrixView&`
// referring to the SAME object -- proving the const overload sees the
// writes the non-const overload made. Finally the whole backing array is
// printed so any write that landed outside the intended 3x3 block (a canary
// getting clobbered) is visible too.
int main() {
    double buf[24];
    for (int i = 0; i < 24; i++) buf[i] = 100.0 + i;  // canary pattern

    MatrixView view(buf + 7, 3, 3, 6);  // 3x3 sub-block, row_stride = 6

    for (long r = 0; r < 3; r++)
        for (long c = 0; c < 3; c++)
            view(r, c) = (double)(r * 3 + c + 1) * 10.0;  // 10.0 .. 90.0

    const MatrixView& cview = view;
    for (long r = 0; r < 3; r++) {
        for (long c = 0; c < 3; c++) {
            printf("%.1f ", cview(r, c));
        }
    }
    printf("\n");

    for (int i = 0; i < 24; i++) printf("%.1f ", buf[i]);
    printf("\n");
    return 0;
}
