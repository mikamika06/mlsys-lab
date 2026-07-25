#include "sol.hpp"
#include <unordered_map>
#include <unordered_set>

long max_reuse_distance(const long* addrs, int n) {
    std::unordered_map<long, int> last_index;  // line -> most recent index touching it
    long best = 0;

    for (int i = 0; i < n; i++) {
        long line = addrs[i] / LINE_BYTES;
        auto it = last_index.find(line);
        if (it != last_index.end()) {
            int j = it->second;
            std::unordered_set<long> distinct;
            for (int k = j + 1; k < i; k++) {
                distinct.insert(addrs[k] / LINE_BYTES);
            }
            long d = (long)distinct.size();
            if (d > best) best = d;
        }
        last_index[line] = i;
    }
    return best;
}
