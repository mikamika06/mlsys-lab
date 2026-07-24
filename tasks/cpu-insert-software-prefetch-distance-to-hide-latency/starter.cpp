#include "sol.hpp"

int count_stalls(int n, int distance, int latency_cycles, int cycles_per_iter) {
    (void)latency_cycles;
    (void)cycles_per_iter;
    // your code here
    return distance < n ? distance : n;
}
