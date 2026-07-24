#include <cstdio>
#include "sol.hpp"

// FIXED driver. Fills three arrays (5, 3, 4 elements) of the three record
// types with deterministic values, calls the learner's in-place mutator on
// each, then prints EVERY field of every element afterward — doubles must
// be +1.0 from their seed value, everything else must be untouched.

constexpr int NA = 5, NB = 3, NC = 4;

int main() {
    RecordA a[NA];
    for (int i = 0; i < NA; ++i) {
        a[i].c = 0;
        a[i].d1 = static_cast<double>(i + 1);
        a[i].d2 = static_cast<double>(i + 2);
    }

    RecordB b[NB];
    for (int i = 0; i < NB; ++i) {
        b[i].d = static_cast<double>(i);
        b[i].c = 1;
        b[i].i = 20;
        b[i].d2 = static_cast<double>(i + 3);
    }

    RecordC c[NC];
    for (int i = 0; i < NC; ++i) {
        c[i].i = 0;
        c[i].d = static_cast<double>(i + 1);
        c[i].f = static_cast<float>(i + 2);
        c[i].d2 = static_cast<double>(i + 3);
    }

    mutate_a(a, NA);
    mutate_b(b, NB);
    mutate_c(c, NC);

    for (int i = 0; i < NA; ++i) {
        printf("%d %.6f %.6f\n", static_cast<int>(a[i].c), a[i].d1, a[i].d2);
    }
    for (int i = 0; i < NB; ++i) {
        printf("%.6f %d %d %.6f\n", b[i].d, static_cast<int>(b[i].c), b[i].i, b[i].d2);
    }
    for (int i = 0; i < NC; ++i) {
        printf("%d %.6f %.6f %.6f\n", c[i].i, c[i].d, static_cast<double>(c[i].f), c[i].d2);
    }
    return 0;
}
