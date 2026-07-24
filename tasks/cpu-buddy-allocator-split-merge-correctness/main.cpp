#include <cstdio>
#include "sol.hpp"

int main() {
    BuddyAllocator bud;

    int a = bud.alloc(20);   // needs a 32-byte block -> splits 256->128->64->32
    int b = bud.alloc(10);   // needs a 16-byte block
    int c = bud.alloc(50);   // needs a 64-byte block
    printf("alloc a=%d b=%d c=%d\n", a, b, c);

    bud.free(a);
    int d = bud.alloc(16);   // should reuse a's freed space (or its buddy)
    printf("after_free_a alloc_d=%d\n", d);

    bud.free(b);
    bud.free(d);
    // a, b, d are all free now; if they (and their buddies) correctly
    // merge back together, plus c's 64-byte block, there still isn't a
    // full 256-byte free block (c is still live) -- but there SHOULD be
    // enough merged space for a 128-byte allocation.
    int e = bud.alloc(100);  // needs a 128-byte block
    printf("after_free_bd alloc_e=%d\n", e);

    int f = bud.alloc(256);  // arena is NOT fully free (c, e still live) -- must fail
    printf("oversized_alloc_f=%d\n", f);

    bud.free(c);
    bud.free(e);
    // everything is free again; if merges chained all the way back up,
    // the whole arena is one free block again
    int g = bud.alloc(256);
    printf("after_free_all alloc_g=%d\n", g);

    return 0;
}
