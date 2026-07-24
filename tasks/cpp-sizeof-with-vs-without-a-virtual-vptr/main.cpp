#include <cstdio>
#include "sol.hpp"

// PROVIDED. Five plain aggregates and their real polymorphic twins (same
// fields, plus a virtual destructor) -- both compiled for real, so
// `sizeof(Virtual1)` etc. below are genuine compiler-computed ground
// truth, not a hand-typed answer.
struct Plain1 { int a; char b; };
struct Virtual1 { virtual ~Virtual1() {} int a; char b; };

struct Plain2 { char a; };
struct Virtual2 { virtual ~Virtual2() {} char a; };

struct Plain3 { double a; int b; char c; };
struct Virtual3 { virtual ~Virtual3() {} double a; int b; char c; };

struct Plain4 { long a; long b; };
struct Virtual4 { virtual ~Virtual4() {} long a; long b; };

struct Plain5 { char a; double b; };
struct Virtual5 { virtual ~Virtual5() {} char a; double b; };

// FIXED driver. Do not edit. For each of the five field layouts, prints
// the plain sizeof/alignof, the learner's derived polymorphic sizeof,
// and the REAL polymorphic struct's sizeof (from the compiler) so both
// numbers are visible.
int main() {
    printf("plain=%d align=%d computed=%ld actual=%d\n",
           (int)sizeof(Plain1), (int)alignof(Plain1),
           virtual_sizeof((long)sizeof(Plain1), (long)alignof(Plain1)),
           (int)sizeof(Virtual1));
    printf("plain=%d align=%d computed=%ld actual=%d\n",
           (int)sizeof(Plain2), (int)alignof(Plain2),
           virtual_sizeof((long)sizeof(Plain2), (long)alignof(Plain2)),
           (int)sizeof(Virtual2));
    printf("plain=%d align=%d computed=%ld actual=%d\n",
           (int)sizeof(Plain3), (int)alignof(Plain3),
           virtual_sizeof((long)sizeof(Plain3), (long)alignof(Plain3)),
           (int)sizeof(Virtual3));
    printf("plain=%d align=%d computed=%ld actual=%d\n",
           (int)sizeof(Plain4), (int)alignof(Plain4),
           virtual_sizeof((long)sizeof(Plain4), (long)alignof(Plain4)),
           (int)sizeof(Virtual4));
    printf("plain=%d align=%d computed=%ld actual=%d\n",
           (int)sizeof(Plain5), (int)alignof(Plain5),
           virtual_sizeof((long)sizeof(Plain5), (long)alignof(Plain5)),
           (int)sizeof(Virtual5));
    return 0;
}
