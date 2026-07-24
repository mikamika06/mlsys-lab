#include <algorithm>
#include <vector>
#include "sol.hpp"

namespace {

// Memoized DFS: finish[i] = latency[i] + max(finish[d] for d in deps[i]),
// or just latency[i] if deps[i] is empty. Works for any DAG (deps[i] may
// reference indices in any order, not just i-1, i-2, ...) as long as there
// is no cycle.
long long compute_finish(int i, const std::vector<int>& latency,
                          const std::vector<std::vector<int>>& deps,
                          std::vector<long long>& finish, std::vector<bool>& done) {
    if (done[i]) return finish[i];
    long long best_pred = 0;
    for (int d : deps[i]) {
        best_pred = std::max(best_pred, compute_finish(d, latency, deps, finish, done));
    }
    finish[i] = best_pred + latency[i];
    done[i] = true;
    return finish[i];
}

}  // namespace

double idealized_ipc(const std::vector<int>& latency,
                      const std::vector<std::vector<int>>& deps,
                      int width) {
    int n = static_cast<int>(latency.size());
    std::vector<long long> finish(n, 0);
    std::vector<bool> done(n, false);

    long long critical_path = 0;
    for (int i = 0; i < n; ++i) {
        critical_path = std::max(critical_path, compute_finish(i, latency, deps, finish, done));
    }

    double width_bound = static_cast<double>(n) / static_cast<double>(width);
    double ideal_cycles = std::max(static_cast<double>(critical_path), width_bound);
    return static_cast<double>(n) / ideal_cycles;
}
