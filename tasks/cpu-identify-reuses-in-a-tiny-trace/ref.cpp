#include "sol.hpp"
#include <unordered_set>

// Reference: a straightforward "have we seen this address before" scan,
// tracking every distinct address ever touched in an unordered_set. No
// notion of cache capacity or eviction -- pure temporal-reuse counting.
long long count_reuses(const long* addrs, int n) {
    std::unordered_set<long> seen;
    long long reuses = 0;
    for (int i = 0; i < n; i++) {
        if (seen.count(addrs[i])) {
            reuses++;
        } else {
            seen.insert(addrs[i]);
        }
    }
    return reuses;
}
