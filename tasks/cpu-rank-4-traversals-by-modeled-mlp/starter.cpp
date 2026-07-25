#include "sol.hpp"

// TODO: process n_misses misses in order. Miss i's ready time is the
// previous miss's completion time if `chained`, else 0. Assign it to
// whichever of the `window` slots frees up earliest; its issue time is
// max(ready, that slot's free time), it completes `latency` cycles
// later. cycles = last completion time; mlp_x1000 = n_misses * latency
// * 1000 / cycles.
MlpResult simulate_mlp(int n_misses, int window, int latency, bool chained) {
    (void)n_misses;
    (void)window;
    (void)latency;
    (void)chained;
    // your code here
    return MlpResult{0, 0};
}
