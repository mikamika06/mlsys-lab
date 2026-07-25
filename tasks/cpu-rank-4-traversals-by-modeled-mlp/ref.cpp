#include "sol.hpp"
#include <vector>

MlpResult simulate_mlp(int n_misses, int window, int latency, bool chained) {
    if (window < 1) window = 1;
    std::vector<long long> slot_free(window, 0);

    long long prev_complete = 0;
    long long max_complete = 0;

    for (int i = 0; i < n_misses; i++) {
        long long ready = chained ? prev_complete : 0;

        int best = 0;
        for (int k = 1; k < window; k++) {
            if (slot_free[k] < slot_free[best]) best = k;
        }

        long long issue = ready > slot_free[best] ? ready : slot_free[best];
        long long complete = issue + latency;
        slot_free[best] = complete;
        prev_complete = complete;
        if (complete > max_complete) max_complete = complete;
    }

    long long cycles = max_complete;
    long long mlp_x1000 = (cycles > 0)
        ? ((long long)n_misses * latency * 1000) / cycles
        : 0;
    return MlpResult{cycles, mlp_x1000};
}
