#include "sol.hpp"

// TODO: return the 4-bit mask of outcomes (r1,r2) observable in the store-buffering
// test. Under seq_cst (store_buffering == false) one outcome is forbidden; under the
// relaxed store-buffering relaxation (store_buffering == true) it becomes observable.
int allowed_outcomes(bool store_buffering) {
    (void)store_buffering;
    return 0;   // your code here
}

// TODO: fill counts[i] with the number of sequentially-consistent interleavings
// (6 in total) that produce outcome i, where i == (r1 << 1) | r2.
void sc_outcome_histogram(int counts[4]) {
    for (int i = 0; i < 4; i++) counts[i] = 0;   // your code here
}
