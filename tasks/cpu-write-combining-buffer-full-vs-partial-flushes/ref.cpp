#include "sol.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace {
struct Entry {
    std::vector<char> touched;
    int count = 0;
};
}

void wc_flush_stats(const long* addrs, int n, int line_bytes, int slots, long* out) {
    std::unordered_map<long, Entry> entries;
    std::vector<long> order; // FIFO: oldest-touched line first
    long full_flush = 0, partial_flush = 0;

    auto flush_line = [&](long line) {
        Entry& e = entries[line];
        if (e.count == line_bytes) full_flush++;
        else partial_flush++;
        entries.erase(line);
        order.erase(std::find(order.begin(), order.end(), line));
    };

    for (int idx = 0; idx < n; idx++) {
        long a = addrs[idx];
        long line = a / line_bytes;
        int offset = (int)(a % line_bytes);

        if (entries.find(line) == entries.end()) {
            if ((int)entries.size() == slots) {
                flush_line(order.front());
            }
            Entry e;
            e.touched.assign(line_bytes, 0);
            entries[line] = std::move(e);
            order.push_back(line);
        }

        Entry& e = entries[line];
        if (!e.touched[offset]) {
            e.touched[offset] = 1;
            e.count++;
        }
        if (e.count == line_bytes) {
            flush_line(line);
        }
    }

    for (long line : std::vector<long>(order)) {
        flush_line(line);
    }

    out[0] = full_flush;
    out[1] = partial_flush;
}
