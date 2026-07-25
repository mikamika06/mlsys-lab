#include <set>
#include "sol.hpp"

long max_working_set_bytes(const long* addrs, int n, int line_bytes, int W) {
    long best = 0;
    for (int t = 0; t + W <= n; t++) {
        std::set<long> lines;
        for (int i = t; i < t + W; i++) lines.insert(addrs[i] / line_bytes);
        long bytes = (long)lines.size() * line_bytes;
        if (bytes > best) best = bytes;
    }
    return best;
}
