#include <algorithm>
#include "sol.hpp"

int dependency_chain_length(const std::vector<int>& depends_on) {
    int n = static_cast<int>(depends_on.size());
    std::vector<int> chain(n, 0);
    int best = 0;
    for (int i = 0; i < n; ++i) {
        int p = depends_on[i];
        chain[i] = 1 + (p == -1 ? 0 : chain[p]);
        best = std::max(best, chain[i]);
    }
    return best;
}
