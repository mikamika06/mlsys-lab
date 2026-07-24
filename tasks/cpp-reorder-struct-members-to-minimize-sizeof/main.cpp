#include <cstdio>
#include "sol.hpp"

// PROVIDED (ground truth): for each case, a real "naive" struct (the given
// field order) and a real "opt" struct (fields sorted descending by size --
// the provably-minimal arrangement for fields whose alignment equals their
// size). sizeof() on these is real, compiler-computed truth, not a
// simulation.
struct A_naive { char a; int b; char c; double d; };
struct A_opt   { double a; int b; char c; char d; };

struct B_naive { short a; char b; short c; char d; double e; };
struct B_opt   { double a; short b; short c; char d; char e; };

struct C_naive { char a; char b; char c; char d; int e; int f; };
struct C_opt   { int a; int b; char c; char d; char e; char f; };

struct D_naive { double a; char b; short c; int d; char e; };
struct D_opt   { double a; int b; short c; char d; char e; };

struct Case {
    const char* name;
    const int* sizes;
    int n;
    int naive_sizeof;
    int opt_sizeof;
};

// FIXED driver. minimal_sizeof() only ever sees the field BYTE SIZES (never
// the struct types themselves) and must reproduce the real compiler's
// sizeof() for the best possible reordering.
int main() {
    static const int szA[] = {1, 4, 1, 8};
    static const int szB[] = {2, 1, 2, 1, 8};
    static const int szC[] = {1, 1, 1, 1, 4, 4};
    static const int szD[] = {8, 1, 2, 4, 1};

    const Case cases[] = {
        {"A", szA, 4, (int)sizeof(A_naive), (int)sizeof(A_opt)},
        {"B", szB, 5, (int)sizeof(B_naive), (int)sizeof(B_opt)},
        {"C", szC, 6, (int)sizeof(C_naive), (int)sizeof(C_opt)},
        {"D", szD, 5, (int)sizeof(D_naive), (int)sizeof(D_opt)},
    };

    for (const auto& c : cases) {
        int predicted = minimal_sizeof(c.sizes, c.n);
        printf("%s naive=%d predicted=%d real_opt=%d\n", c.name, c.naive_sizeof, predicted, c.opt_sizeof);
    }
    return 0;
}
