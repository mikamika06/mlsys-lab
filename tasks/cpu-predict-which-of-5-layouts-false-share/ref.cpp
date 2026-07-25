#include "sol.hpp"
#include <vector>

static std::vector<long> layout_addrs(int layout_id, int num_threads) {
    std::vector<long> addrs;
    for (int t = 0; t < num_threads; t++) {
        long a = 0;
        switch (layout_id) {
            case 0: a = (long)t * 8; break;
            case 1: a = (long)t * 64; break;
            case 2: a = (long)t * 128; break;
            case 3: a = (long)t * 8 + 64 * (t % 2); break;
            case 4: a = (long)t * 16; break;
        }
        addrs.push_back(a);
    }
    return addrs;
}

static bool has_false_sharing(const std::vector<long>& addrs, long line_bytes) {
    std::vector<long> lines;
    for (long a : addrs) lines.push_back(a / line_bytes);
    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (lines[i] == lines[j]) return true;
        }
    }
    return false;
}

std::array<bool, 5> classify_layouts(long line_bytes) {
    const int num_threads = 4;
    std::array<bool, 5> out{};
    for (int k = 0; k < 5; k++) {
        out[k] = has_false_sharing(layout_addrs(k, num_threads), line_bytes);
    }
    return out;
}
