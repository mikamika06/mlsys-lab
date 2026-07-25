// Fixed driver: a fixed table of (latency_cycles, cycles_per_iter,
// mlp_max) triples spanning both regimes -- latency-bound (MLP headroom
// to spare) and MLP-bound (hardware's outstanding-request limit binds
// first). No timing, no randomness -- fixed integer inputs throughout.
#include "sol.hpp"
#include <cstdio>

namespace {
struct Case { int latency_cycles, cycles_per_iter, mlp_max; };

const Case CASES[] = {
    {180, 20, 16},   // required=9,  mlp_max=16 -> latency-bound, fully hidden
    {180, 20, 6},    // required=9,  mlp_max=6  -> MLP-bound, not fully hidden
    {400, 25, 10},   // required=16, mlp_max=10 -> MLP-bound
    {100, 50, 10},   // required=2,  mlp_max=10 -> latency-bound, fully hidden
    {250, 10, 32},   // required=25, mlp_max=32 -> latency-bound, fully hidden
    {500, 8, 12},    // required=63, mlp_max=12 -> MLP-bound
};
const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);
} // namespace

int main() {
    for (int i = 0; i < NUM_CASES; i++) {
        const Case& c = CASES[i];
        int d = min_saturating_distance(c.latency_cycles, c.cycles_per_iter, c.mlp_max);
        bool hidden = is_latency_fully_hidden(c.latency_cycles, c.cycles_per_iter, c.mlp_max);
        printf("L=%d C=%d mlp_max=%d distance=%d hidden=%s\n", c.latency_cycles, c.cycles_per_iter,
               c.mlp_max, d, hidden ? "true" : "false");
    }
    return 0;
}
