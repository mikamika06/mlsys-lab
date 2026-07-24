#include <set>
#include "sol.hpp"

long count_distinct_lines(const long* addrs, int n, int line_bytes) {
    std::set<long> lines;
    for (int i = 0; i < n; i++) lines.insert(addrs[i] / line_bytes);
    return (long)lines.size();
}
