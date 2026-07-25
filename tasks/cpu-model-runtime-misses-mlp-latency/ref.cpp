#include "sol.hpp"

double modeled_cycles(long num_misses, int mlp, double miss_latency) {
    return ((double)num_misses / (double)mlp) * miss_latency;
}
