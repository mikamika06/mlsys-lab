#include <cstdio>
#include <climits>
#include "sol.hpp"

// FIXED driver. Both functions are compiled by the SAME clang++ -O2
// invocation as the rest of this file; `check_no_overflow_noopt` is forced
// unoptimized via `__attribute__((optnone))` in sol.hpp, giving a REAL
// "what does -O0 do" data point without a second compiler invocation.
int main() {
    const int values[] = {0, -1, INT_MIN, INT_MAX - 1, INT_MAX};
    const int NV = 5;

    for (int i = 0; i < NV; i++) {
        int x = values[i];
        bool r_noopt = check_no_overflow_noopt(x);
        bool r_opt = check_no_overflow_opt(x);
        printf("x=%d noopt=%d opt=%d agree=%d\n", x, r_noopt ? 1 : 0, r_opt ? 1 : 0, (r_noopt == r_opt) ? 1 : 0);
    }
    return 0;
}
