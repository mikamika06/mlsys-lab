#include <cstdint>
#include <cstdio>

#include "sol.hpp"

Tracked operator+(Tracked a, Tracked b) {
    Tracked r;
    r.value = a.value + b.value;
    r.depth = (a.depth > b.depth ? a.depth : b.depth) + 1;
    return r;
}

static void fill(Tracked* x, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t h = (uint32_t)i * 1103515245u + 12345u;
        x[i].value = (double)((int)((h >> 8) & 1023u) - 512);  // integer in [-512, 511]
        x[i].depth = 0;
    }
}

// FIXED driver: two deterministic integer-valued fixtures (so sums are
// exact regardless of association order) -- N1 = 4096 (a power of two,
// so a perfectly balanced tree has EXACT depth log2(4096) = 12) and
// N2 = 1000 (not a power of two, exact depth ceil(log2(1000)) = 10, the
// standard divide-and-conquer height identity). Prints both sums and
// both resulting depths.
int main() {
    const int N1 = 4096;
    static Tracked x1[N1];
    fill(x1, N1);
    Tracked t1 = reduce_balanced_tree(x1, N1);

    const int N2 = 1000;
    static Tracked x2[N2];
    fill(x2, N2);
    Tracked t2 = reduce_balanced_tree(x2, N2);

    printf("%.6f depth=%d\n", t1.value, t1.depth);
    printf("%.6f depth=%d\n", t2.value, t2.depth);
    return 0;
}
