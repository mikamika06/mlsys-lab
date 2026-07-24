#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. Every DAG below is hand-built (no rand()/time()) so the
// expected numbers are easy to hand-verify (see task.md).

namespace {

void run(const char* name, const std::vector<int>& latency,
         const std::vector<std::vector<int>>& deps, int width) {
    double ipc = idealized_ipc(latency, deps, width);
    printf("%s n=%zu width=%d ipc=%.6f\n", name, latency.size(), width, ipc);
}

}  // namespace

int main() {
    // A: 16 fully independent 1-cycle instructions, width=4.
    // No dependencies at all -> bound entirely by issue width.
    {
        int n = 16;
        std::vector<int> latency(n, 1);
        std::vector<std::vector<int>> deps(n);
        run("A", latency, deps, 4);
    }

    // B: a pure serial chain of 10 1-cycle instructions, width=4.
    // Every instruction depends on the previous one -> bound entirely by
    // the critical path, no matter how wide the machine is.
    {
        int n = 10;
        std::vector<int> latency(n, 1);
        std::vector<std::vector<int>> deps(n);
        for (int i = 1; i < n; ++i) deps[i] = {i - 1};
        run("B", latency, deps, 4);
    }

    // C: a mixed DAG, width=4.
    //   chain 0->1->2->3, latency 2 each              (critical contribution 8)
    //   chain 4->5, latency 3 each                     (critical contribution 6)
    //   6 independent 1-cycle instructions (6..11)     (critical contribution 1)
    // n=12, critical_path=8, width_bound=12/4=3.0 -> ideal_cycles=8, ipc=1.5
    {
        std::vector<int> latency = {2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1};
        std::vector<std::vector<int>> deps(12);
        deps[1] = {0};
        deps[2] = {1};
        deps[3] = {2};
        deps[5] = {4};
        run("C", latency, deps, 4);
    }

    // D: the SAME DAG as C, but width=1.
    // critical_path is still 8 (dependencies don't change), but
    // width_bound=12/1=12.0 now dominates -> ideal_cycles=12, ipc=1.0
    {
        std::vector<int> latency = {2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1};
        std::vector<std::vector<int>> deps(12);
        deps[1] = {0};
        deps[2] = {1};
        deps[3] = {2};
        deps[5] = {4};
        run("D", latency, deps, 1);
    }

    return 0;
}
