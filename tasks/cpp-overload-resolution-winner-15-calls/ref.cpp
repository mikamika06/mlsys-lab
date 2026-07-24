#include "sol.hpp"

// Reference predictions for the 15 scenarios documented in task.md,
// matching real C++ overload resolution.
void predict_overload_winners(int out[15]) {
    int labels[15] = {
        /* 1  */ 0,
        /* 2  */ 1,
        /* 3  */ 0,
        /* 4  */ 1,
        /* 5  */ 1,
        /* 6  */ 1,
        /* 7  */ 0,
        /* 8  */ 0,
        /* 9  */ 0,
        /* 10 */ 1,
        /* 11 */ 0,
        /* 12 */ 0,
        /* 13 */ 0,
        /* 14 */ 1,
        /* 15 */ 0,
    };
    for (int i = 0; i < 15; i++) out[i] = labels[i];
}
