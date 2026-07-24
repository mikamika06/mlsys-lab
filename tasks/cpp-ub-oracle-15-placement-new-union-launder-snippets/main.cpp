#include <cstdio>
#include "sol.hpp"

// FIXED driver. Eleven traces, each hand-picked to isolate one rule (or
// combination) from sol.hpp: a stale pointer after reusing const storage,
// laundering fixing it (permanently, not just for one access), writing
// through a const object, overwriting a non-trivial object without its
// destructor, a proper dtor-then-reuse that IS fine, accessing dead
// storage, a double-destroy, and a type-tag mismatch.
int main() {
    static const Op a[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 1, 1, 0, 0},
        {PLACEMENT_NEW, 1, 0, 1, 0, 0},
        {ACCESS, 1, 0, 0, 0, 0},
    };
    static const Op b[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 1, 1, 0, 0},
        {PLACEMENT_NEW, 1, 0, 1, 0, 0},
        {ACCESS, 1, 0, 0, 0, 1},
    };
    static const Op c[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 0, 1, 0, 0},
        {ACCESS, 0, 0, 0, 1, 0},
    };
    static const Op d[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 1, 1, 0, 0},
        {ACCESS, 0, 0, 0, 1, 0},
    };
    static const Op e[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 2, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 0, 1, 0, 0},
    };
    static const Op f[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 2, 0, 0, 0, 0},
        {DTOR, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 0, 1, 0, 0},
        {ACCESS, 0, 0, 0, 0, 0},
    };
    static const Op g[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {ACCESS, 0, 0, 0, 0, 0},
    };
    static const Op h[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 0, 1, 0, 0},
        {DTOR, 0, 0, 0, 0, 0},
        {DTOR, 0, 0, 0, 0, 0},
    };
    static const Op i[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 1, 0, 1, 0, 0},
        {ACCESS, 0, 0, 0, 0, 0},
    };
    static const Op j[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 1, 1, 0, 0},
        {DTOR, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 1, 0, 1, 0, 0},
        {ACCESS, 1, 0, 0, 0, 0},
    };
    static const Op k[] = {
        {ALLOCATE, 0, 0, 0, 0, 0},
        {PLACEMENT_NEW, 0, 1, 1, 0, 0},
        {PLACEMENT_NEW, 1, 0, 1, 0, 0},
        {ACCESS, 1, 0, 0, 0, 1},
        {ACCESS, 1, 0, 0, 0, 0},
    };

    struct Case { const char* name; const Op* ops; int n; };
    const Case cases[] = {
        {"a", a, (int)(sizeof(a) / sizeof(a[0]))},
        {"b", b, (int)(sizeof(b) / sizeof(b[0]))},
        {"c", c, (int)(sizeof(c) / sizeof(c[0]))},
        {"d", d, (int)(sizeof(d) / sizeof(d[0]))},
        {"e", e, (int)(sizeof(e) / sizeof(e[0]))},
        {"f", f, (int)(sizeof(f) / sizeof(f[0]))},
        {"g", g, (int)(sizeof(g) / sizeof(g[0]))},
        {"h", h, (int)(sizeof(h) / sizeof(h[0]))},
        {"i", i, (int)(sizeof(i) / sizeof(i[0]))},
        {"j", j, (int)(sizeof(j) / sizeof(j[0]))},
        {"k", k, (int)(sizeof(k) / sizeof(k[0]))},
    };

    for (int idx = 0; idx < 11; idx++) {
        printf("%s=%d\n", cases[idx].name, classify_ub(cases[idx].ops, cases[idx].n));
    }
    return 0;
}
