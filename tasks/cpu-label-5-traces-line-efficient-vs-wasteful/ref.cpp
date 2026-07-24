#include <set>
#include "sol.hpp"

int classify_trace(const long* addrs, int num_accesses, int elem_bytes, int line_bytes) {
    std::set<long> lines;
    for (int i = 0; i < num_accesses; i++) {
        lines.insert(addrs[i] / line_bytes);
    }
    double bytes_used = (double)num_accesses * elem_bytes;
    double bytes_fetched = (double)lines.size() * line_bytes;
    double efficiency = bytes_used / bytes_fetched;
    return efficiency >= 0.5 ? 1 : 0;
}
