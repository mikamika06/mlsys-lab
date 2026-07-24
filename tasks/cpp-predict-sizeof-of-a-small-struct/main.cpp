#include <cstdio>
#include "sol.hpp"

// Eight real 2-3 field structs (this file is the only place that "cheats"
// by asking the real compiler for sizeof() -- it's the ground truth the
// driver cross-checks against, not something predict_sizeof gets to see).
struct S1 { char a; int b; };
struct S2 { int a; char b; };
struct S3 { char a; double b; };
struct S4 { char a; short b; int c; };
struct S5 { double a; char b; };
struct S6 { short a; char b; double c; };
struct S7 { int a; double b; char c; };
struct S8 { char a; char b; long c; };

struct Case {
    const char* name;
    const int* sizes;
    int n;
    int real_sizeof;
};

// FIXED driver. For each struct, predict_sizeof() only ever sees the field
// BYTE SIZES in declaration order -- never the struct type itself -- and
// must reproduce what the real compiler computes for sizeof(SK).
int main() {
    static const int sz1[] = {1, 4};
    static const int sz2[] = {4, 1};
    static const int sz3[] = {1, 8};
    static const int sz4[] = {1, 2, 4};
    static const int sz5[] = {8, 1};
    static const int sz6[] = {2, 1, 8};
    static const int sz7[] = {4, 8, 1};
    static const int sz8[] = {1, 1, 8};

    const Case cases[] = {
        {"S1", sz1, 2, (int)sizeof(S1)},
        {"S2", sz2, 2, (int)sizeof(S2)},
        {"S3", sz3, 2, (int)sizeof(S3)},
        {"S4", sz4, 3, (int)sizeof(S4)},
        {"S5", sz5, 2, (int)sizeof(S5)},
        {"S6", sz6, 3, (int)sizeof(S6)},
        {"S7", sz7, 3, (int)sizeof(S7)},
        {"S8", sz8, 3, (int)sizeof(S8)},
    };

    for (int i = 0; i < 8; i++) {
        int predicted = predict_sizeof(cases[i].sizes, cases[i].n);
        printf("%s predicted=%d real=%d\n", cases[i].name, predicted, cases[i].real_sizeof);
    }
    return 0;
}
