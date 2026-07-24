#include "sol.hpp"

// Reference predictions for the 15 snippets documented in task.md, matching
// what -fsanitize=undefined actually reports.
void predict_ubsan_flags(int out[15]) {
    int labels[15] = {
        /* 1  signed overflow            */ 1,
        /* 2  signed, in range           */ 0,
        /* 3  unsigned overflow (ok)     */ 0,
        /* 4  unsigned multiply wrap(ok) */ 0,
        /* 5  misaligned int*            */ 1,
        /* 6  aligned int*               */ 0,
        /* 7  misaligned double*         */ 1,
        /* 8  shift >= 32                */ 1,
        /* 9  shift in range             */ 0,
        /* 10 negative shift             */ 1,
        /* 11 division by zero           */ 1,
        /* 12 modulo by zero             */ 1,
        /* 13 division, nonzero          */ 0,
        /* 14 array index out of bounds  */ 1,
        /* 15 array index in bounds      */ 0,
    };
    for (int i = 0; i < 15; i++) out[i] = labels[i];
}
