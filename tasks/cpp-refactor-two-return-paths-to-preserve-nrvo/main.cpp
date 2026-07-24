#include <cstdio>
#include "sol.hpp"

// FIXED driver: four (cond, a1, b1, a2, b2) scenarios. Counters are reset
// before each call so every line reports that call's own copy/move count.

static void run_case(bool cond, int a1, double b1, int a2, double b2) {
    g_copy_count = 0;
    g_move_count = 0;
    Result r = make_result(cond, a1, b1, a2, b2);
    printf("%d %.6f %d %d\n", r.a, r.b, g_copy_count, g_move_count);
}

int main() {
    run_case(true, 10, 3.14, 20, 2.718);
    run_case(false, 10, 3.14, 20, 2.718);
    run_case(true, 65, 100000.0, 66, 200000.0);
    run_case(false, 100, 1.23, 200, 4.56);
    return 0;
}
