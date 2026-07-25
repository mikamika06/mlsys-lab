#include "sol.hpp"

namespace {
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
} // namespace

int min_saturating_distance(int latency_cycles, int cycles_per_iter, int mlp_max) {
    int required = ceil_div(latency_cycles, cycles_per_iter);
    return required < mlp_max ? required : mlp_max;
}

bool is_latency_fully_hidden(int latency_cycles, int cycles_per_iter, int mlp_max) {
    int required = ceil_div(latency_cycles, cycles_per_iter);
    return required <= mlp_max;
}
