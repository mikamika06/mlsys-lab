#include <cstdio>
#include "sol.hpp"

// Deterministic MLP model (harness code, not learner code). Independent
// loads round-robin across MAX_INFLIGHT slots (chain ids 0..7); every
// dependent load extends one shared serial chain (chain id
// MAX_INFLIGHT). critical_path_cycles() is the longest chain's length.
namespace {
constexpr int MAX_INFLIGHT = 8;
constexpr int SERIAL_CHAIN = MAX_INFLIGHT;  // chain id reserved for dependent loads
long g_count[MAX_INFLIGHT + 1];
long g_independent_seen;
}  // namespace

void reset_mlp() {
    for (int i = 0; i <= MAX_INFLIGHT; i++) g_count[i] = 0;
    g_independent_seen = 0;
}

void report_load(bool independent) {
    if (independent) {
        int slot = (int)(g_independent_seen % MAX_INFLIGHT);
        g_independent_seen++;
        g_count[slot]++;
    } else {
        g_count[SERIAL_CHAIN]++;
    }
}

long critical_path_cycles() {
    long m = 0;
    for (int i = 0; i <= MAX_INFLIGHT; i++) {
        if (g_count[i] > m) m = g_count[i];
    }
    return m;
}

// FIXED driver. 50-node linked list; the chain visits nodes in the order
// order[k] = (k * 13) % 50 (13 and 50 are coprime, so this is a single
// permutation cycle touching every node once) -- built here with real
// index arithmetic, not a hardcoded list.
int main() {
    const int n = 50;
    int order[n];
    for (int k = 0; k < n; k++) order[k] = (k * 13) % n;

    int next_idx[n];
    for (int k = 0; k < n - 1; k++) next_idx[order[k]] = order[k + 1];
    next_idx[order[n - 1]] = -1;
    int head = order[0];

    double values[n];
    for (int i = 0; i < n; i++) values[i] = (double)((i * 31 % 97) - 48);

    int traversal[n];
    for (int i = 0; i < n; i++) traversal[i] = -1;  // sentinel
    reset_mlp();
    pointer_chase_traversal(next_idx, head, n, traversal);
    long chase_critical_path = critical_path_cycles();

    double gathered[n];
    for (int i = 0; i < n; i++) gathered[i] = -999.0;  // sentinel
    reset_mlp();
    gather_by_index(values, traversal, n, gathered);
    long gather_critical_path = critical_path_cycles();

    printf("chase_critical_path=%ld gather_critical_path=%ld\n", chase_critical_path, gather_critical_path);
    for (int i = 0; i < n; i++) printf("%d ", traversal[i]);
    printf("\n");
    for (int i = 0; i < n; i++) printf("%.1f ", gathered[i]);
    printf("\n");
    return 0;
}
