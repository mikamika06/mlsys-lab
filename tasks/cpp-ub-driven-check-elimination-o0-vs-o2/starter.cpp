#include "sol.hpp"

// BUG: relies on signed overflow "wrapping around" to detect it. Signed
// overflow is undefined behavior, so at real optimization levels the
// compiler is free to assume `x + 1 < x` never happens and delete the
// check -- which is exactly what happens to check_no_overflow_opt below,
// even though check_no_overflow_noopt (forced unoptimized via sol.hpp's
// __attribute__((optnone))) still evaluates it literally.
__attribute__((optnone)) bool check_no_overflow_noopt(int x) {
    int y = x + 1;
    if (y < x) return false;
    return true;
}

bool check_no_overflow_opt(int x) {
    int y = x + 1;
    if (y < x) return false;
    return true;
}
