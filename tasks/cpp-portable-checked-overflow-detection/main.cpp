#include <cstdio>
#include <climits>
#include "sol.hpp"

// FIXED driver: the twelve scenarios from the task, each calling exactly
// the (op, type) predicate it needs. Prints each result as 0/1.

int main() {
    bool r[12];
    r[0]  = add_overflow_int(2000000000, 1500000000);
    r[1]  = add_overflow_int(-2000000000, 1500000000);
    r[2]  = mul_overflow_short(300, 300);
    r[3]  = add_overflow_char(100, 50);
    r[4]  = sub_overflow_char(-100, 50);
    r[5]  = mul_overflow_char(10, 15);
    r[6]  = add_overflow_long(9000000000000000000L, 1000000000000000000L);
    r[7]  = sub_overflow_int(INT_MIN, 1);
    r[8]  = mul_overflow_int(100000, 100000);
    r[9]  = mul_overflow_long(100000L, 100000L);
    r[10] = add_overflow_short(32000, 1000);
    r[11] = sub_overflow_short(-32000, 1000);

    for (int i = 0; i < 12; ++i) printf("%d ", r[i] ? 1 : 0);
    printf("\n");
    return 0;
}
