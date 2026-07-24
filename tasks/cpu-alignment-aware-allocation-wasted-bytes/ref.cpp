#include "sol.hpp"

// Reference: real bump-allocator alignment math. Since every alignment is
// a power of two, `(offset + a - 1) & ~(a - 1)` rounds offset up to the
// next multiple of a -- the same bit trick a real allocator uses.
static long align_up(long offset, long a) {
    return (offset + a - 1) & ~(a - 1);
}

long total_wasted_bytes(const int* sizes, const int* alignments, int n) {
    long offset = 0;
    long wasted = 0;
    for (int i = 0; i < n; i++) {
        long aligned = align_up(offset, alignments[i]);
        wasted += aligned - offset;
        offset = aligned + sizes[i];
    }
    return wasted;
}
