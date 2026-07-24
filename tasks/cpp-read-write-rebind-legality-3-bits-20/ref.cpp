#include "sol.hpp"

// Reference predictions (may_read, may_write_through, may_rebind) for each
// of the 20 declarations documented in task.md, matching real C++
// const-correctness rules.
void predict_legality(int out[60]) {
    int labels[20][3] = {
        /* 0  int* p                          */ {1, 1, 1},
        /* 1  const int* p                    */ {1, 0, 1},
        /* 2  int* const p                    */ {1, 1, 0},
        /* 3  const int* const p              */ {1, 0, 0},
        /* 4  int& r                          */ {1, 1, 0},
        /* 5  const int& r                    */ {1, 0, 0},
        /* 6  double* p                       */ {1, 1, 1},
        /* 7  const double* p                 */ {1, 0, 1},
        /* 8  double* const p                 */ {1, 1, 0},
        /* 9  char& r                         */ {1, 1, 0},
        /* 10 const char& r                   */ {1, 0, 0},
        /* 11 void* p                         */ {0, 0, 1},
        /* 12 const void* p                   */ {0, 0, 1},
        /* 13 void* const p                   */ {0, 0, 0},
        /* 14 int*& rp                        */ {1, 1, 0},
        /* 15 int* const& rp                  */ {1, 1, 0},
        /* 16 const int*& rp                  */ {1, 0, 0},
        /* 17 const int* const& rp            */ {1, 0, 0},
        /* 18 long* const p                   */ {1, 1, 0},
        /* 19 const long* p                   */ {1, 0, 1},
    };
    for (int i = 0; i < 20; i++) {
        out[i * 3 + 0] = labels[i][0];
        out[i * 3 + 1] = labels[i][1];
        out[i * 3 + 2] = labels[i][2];
    }
}
