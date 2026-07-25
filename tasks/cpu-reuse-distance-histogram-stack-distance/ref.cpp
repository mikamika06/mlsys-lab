#include "sol.hpp"
#include <vector>
#include <set>

void stack_distance_histogram(const long* addrs, int n, int line_bytes,
                               int num_lines, long* hist_out) {
    for (int i = 0; i <= num_lines; ++i) hist_out[i] = 0;

    std::vector<long> line(n);
    for (int i = 0; i < n; ++i) line[i] = addrs[i] / line_bytes;

    for (int i = 0; i < n; ++i) {
        int prev = -1;
        for (int j = i - 1; j >= 0; --j) {
            if (line[j] == line[i]) {
                prev = j;
                break;
            }
        }
        if (prev == -1) {
            hist_out[0] += 1;
            continue;
        }
        std::set<long> distinct;
        for (int k = prev + 1; k < i; ++k) distinct.insert(line[k]);
        int d = static_cast<int>(distinct.size());
        hist_out[1 + d] += 1;
    }
}
