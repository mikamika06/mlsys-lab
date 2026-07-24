#include <cstdio>
#include <vector>
#include "sol.hpp"

// FIXED driver. 4 hand-built access patterns, all over N=16 accesses (no
// rand()/time()), spanning the two extremes (one long pointer-chase, N
// fully independent loads) and two intermediate cases (2 and 4 interleaved
// chains -- K independent pointer chases sharing one access stream).

namespace {

void run(const char* name, const std::vector<int>& depends_on) {
    int len = dependency_chain_length(depends_on);
    printf("%s n=%zu chain_length=%d\n", name, depends_on.size(), len);
}

}  // namespace

int main() {
    constexpr int N = 16;

    // pointer_chase: access i's address comes from access i-1's result --
    // a real linked-list traversal. Every access is serialized.
    {
        std::vector<int> deps(N);
        deps[0] = -1;
        for (int i = 1; i < N; ++i) deps[i] = i - 1;
        run("pointer_chase", deps);
    }

    // independent: all N addresses are known up front (e.g. reading
    // array[0..N) directly). None of them waits on any other.
    {
        std::vector<int> deps(N, -1);
        run("independent", deps);
    }

    // two_chains: 2 independent pointer chases, interleaved one step at a
    // time, each 8 accesses long -- half the machine's outstanding
    // requests always belong to a chain, but the two chains can overlap.
    {
        std::vector<int> deps(N);
        for (int i = 0; i < N; ++i) deps[i] = (i < 2) ? -1 : i - 2;
        run("two_chains", deps);
    }

    // four_chains: 4 independent pointer chases, interleaved, each 4
    // accesses long.
    {
        std::vector<int> deps(N);
        for (int i = 0; i < N; ++i) deps[i] = (i < 4) ? -1 : i - 4;
        run("four_chains", deps);
    }

    return 0;
}
