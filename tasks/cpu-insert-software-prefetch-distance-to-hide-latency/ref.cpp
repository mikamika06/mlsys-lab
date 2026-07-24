#include "sol.hpp"

int count_stalls(int n, int distance, int latency_cycles, int cycles_per_iter) {
    int warmup = distance < n ? distance : n;
    if (distance * cycles_per_iter >= latency_cycles) {
        return warmup;
    }
    return n;
}
