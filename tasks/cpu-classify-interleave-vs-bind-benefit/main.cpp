// Fixed driver: pins the NUMA model constants, runs a fixed battery of
// per-node access-count workloads through classify_workload, and prints
// both computed AMATs plus the winning label for each. Determinism is
// by construction: every input is a fixed integer table, no timing, no
// randomness.
#include "sol.hpp"
#include <cstdio>

const double LOCAL_NS = 90.0;
const double REMOTE_NS = 190.0;
const double CONTENTION_COEF_NS = 15.0;

namespace {
struct Case {
    const char* name;
    int num_nodes;
    long counts[8];
};

const Case CASES[] = {
    // One node dominates almost completely -> bind should win big.
    {"owner_heavy_4n",   4, {9700, 100, 100, 100, 0, 0, 0, 0}},
    // Perfectly even across all 4 nodes -> interleave should win.
    {"even_4n",          4, {2500, 2500, 2500, 2500, 0, 0, 0, 0}},
    // Two-node 60/40 split, other two nodes idle -> bind should win.
    {"two_node_split",   4, {6000, 4000, 0, 0, 0, 0, 0, 0}},
    // 8 nodes, one clear dominant owner, rest share a thin remainder.
    {"owner_heavy_8n",   8, {8600, 210, 190, 200, 220, 195, 205, 180}},
    // 8 nodes, near-uniform noisy split -> interleave should win.
    {"even_8n_noisy",    8, {1180, 1250, 1300, 1190, 1260, 1240, 1310, 1270}},
    // 4 nodes, moderate skew (50/20/20/10), no single dominant owner.
    {"moderate_skew_4n", 4, {5000, 2000, 2000, 1000, 0, 0, 0, 0}},
    // 8 nodes, one node owns virtually everything, others near-zero.
    {"single_owner_8n",  8, {9999, 1, 1, 1, 1, 1, 1, 1}},
};
const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);
} // namespace

int main() {
    for (int i = 0; i < NUM_CASES; i++) {
        const Case& c = CASES[i];
        Workload w{c.num_nodes, c.counts};
        double bind_ns = 0.0, interleave_ns = 0.0;
        const char* label = classify_workload(w, &bind_ns, &interleave_ns);
        printf("%-16s bind=%.3f interleave=%.3f label=%s\n", c.name, bind_ns, interleave_ns, label);
    }
    return 0;
}
