#include <cstdio>
#include "sol.hpp"

// Deterministic ILP critical-path model (harness code, not learner
// code). Tracks how many times each chain_id has been reported since
// the last reset; the critical path is whichever chain was reported the
// most.
namespace {
constexpr int MAX_CHAINS = 64;
long g_count[MAX_CHAINS];
int g_max_chain_seen;
}  // namespace

void reset_ilp() {
    for (int i = 0; i < MAX_CHAINS; i++) g_count[i] = 0;
    g_max_chain_seen = -1;
}

void report_op(int chain_id) {
    g_count[chain_id]++;
    if (chain_id > g_max_chain_seen) g_max_chain_seen = chain_id;
}

long critical_path_cycles() {
    long m = 0;
    for (int i = 0; i <= g_max_chain_seen; i++) {
        if (g_count[i] > m) m = g_count[i];
    }
    return m;
}

// FIXED driver. n=97 (not a multiple of num_chains, to exercise
// remainder handling), num_chains=4, deterministic integer-valued a/b.
int main() {
    const int n = 97;
    const int num_chains = 4;
    double a[n], b[n];
    for (int i = 0; i < n; i++) {
        a[i] = (double)((i % 7) - 3);
        b[i] = (double)((i % 5) - 2);
    }

    reset_ilp();
    double value = -1.0;  // sentinel: an empty starter leaves this untouched
    value = dot_product_ilp(a, b, n, num_chains);
    long critical_path = critical_path_cycles();

    printf("value=%.4f critical_path=%ld\n", value, critical_path);
    return 0;
}
