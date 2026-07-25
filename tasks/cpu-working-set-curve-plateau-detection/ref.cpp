#include "sol.hpp"
#include <set>

int plateau_index(const long* addrs, int n, int max_w, int line_bytes, int* curve_out) {
    for (int w = 1; w <= max_w; w++) {
        std::set<long> lines;
        for (int i = n - w; i < n; i++) {
            lines.insert(addrs[i] / line_bytes);
        }
        curve_out[w - 1] = (int)lines.size();
    }

    int final_val = curve_out[max_w - 1];
    for (int w = 1; w <= max_w; w++) {
        if (curve_out[w - 1] == final_val) return w;
    }
    return max_w;
}
