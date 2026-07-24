#include "sol.hpp"

Result make_result(bool cond, int a1, double b1, int a2, double b2) {
    Result r;
    if (cond) {
        r.a = a1;
        r.b = b1;
    } else {
        r.a = a2;
        r.b = b2;
    }
    return r;
}
