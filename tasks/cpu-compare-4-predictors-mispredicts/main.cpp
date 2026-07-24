#include <cstdio>
#include "sol.hpp"

// FIXED driver. Trace: 8 taken, 8 not-taken, then 8 alternating
// taken/not-taken -- a mix that gives each predictor a different
// mispredict count. hist_bits = 2 -> gshare table has 4 entries.
int main() {
    int outcomes[24] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0,
    };
    const int n = 24;
    const int hist_bits = 2;

    int out[4] = {-1, -1, -1, -1};  // sentinel: an empty starter leaves this untouched
    predictor_mispredicts(outcomes, n, hist_bits, out);

    printf("always_taken=%d one_bit=%d two_bit=%d gshare=%d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
