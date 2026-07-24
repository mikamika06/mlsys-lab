#include "sol.hpp"

// Reference predictions (0=nothrow, 1=strong, 2=basic, 3=none) for f1..f12,
// matching the empirical classification the driver computes via real fault
// injection.
void classify_guarantees(int out[12]) {
    int labels[12] = {
        /* f1  */ 0,
        /* f2  */ 2,
        /* f3  */ 1,
        /* f4  */ 2,
        /* f5  */ 3,
        /* f6  */ 1,
        /* f7  */ 2,
        /* f8  */ 1,
        /* f9  */ 1,
        /* f10 */ 2,
        /* f11 */ 0,
        /* f12 */ 3,
    };
    for (int i = 0; i < 12; i++) out[i] = labels[i];
}
