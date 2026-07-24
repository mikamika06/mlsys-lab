#include "sol.hpp"

const char* classify_workload(const Workload& w, double* bind_ns, double* interleave_ns) {
    long total = 0;
    long best = 0;
    for (int k = 0; k < w.num_nodes; k++) {
        total += w.access_count[k];
        if (w.access_count[k] > best) best = w.access_count[k];
    }

    double L = static_cast<double>(best) / static_cast<double>(total);
    double base_bind = L * LOCAL_NS + (1.0 - L) * REMOTE_NS;

    double hhi = 0.0;
    for (int k = 0; k < w.num_nodes; k++) {
        double f = static_cast<double>(w.access_count[k]) / static_cast<double>(total);
        hhi += f * f;
    }
    double effective_sharers = 1.0 / hhi;
    *bind_ns = base_bind + CONTENTION_COEF_NS * (effective_sharers - 1.0);

    double n = static_cast<double>(w.num_nodes);
    *interleave_ns = LOCAL_NS / n + REMOTE_NS * (n - 1.0) / n;

    return (*interleave_ns < *bind_ns) ? "interleave" : "bind";
}
