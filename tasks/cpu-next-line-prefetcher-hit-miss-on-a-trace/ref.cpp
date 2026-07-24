#include <vector>
#include "sol.hpp"

namespace {
int find_pos(const std::vector<int>& lines, int line) {
    for (int i = 0; i < (int)lines.size(); i++)
        if (lines[i] == line) return i;
    return -1;
}
}  // namespace

void simulate_next_line_prefetch(const int* line_trace, int n, int cache_lines,
                                  int* hits_out, int* misses_out) {
    std::vector<int> lines;  // front = MRU .. back = LRU
    int hits = 0, misses = 0;

    for (int i = 0; i < n; i++) {
        int L = line_trace[i];
        int pos = find_pos(lines, L);
        if (pos >= 0) {
            hits++;
            lines.erase(lines.begin() + pos);
            lines.insert(lines.begin(), L);
        } else {
            misses++;
            if ((int)lines.size() == cache_lines) lines.pop_back();
            lines.insert(lines.begin(), L);
        }

        int P = L + 1;
        if (find_pos(lines, P) < 0) {
            if ((int)lines.size() == cache_lines) lines.pop_back();
            lines.push_back(P);
        }
    }

    *hits_out = hits;
    *misses_out = misses;
}
