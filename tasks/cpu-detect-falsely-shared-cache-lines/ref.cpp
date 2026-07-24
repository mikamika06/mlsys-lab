#include <map>
#include <set>
#include "sol.hpp"

int find_falsely_shared_lines(const long* addrs, const int* thread_id, int n, int line_bytes, long* out) {
    std::map<long, std::set<long>> line_addrs;
    std::map<long, std::set<int>> line_threads;

    for (int i = 0; i < n; i++) {
        long line = addrs[i] / line_bytes;
        line_addrs[line].insert(addrs[i]);
        line_threads[line].insert(thread_id[i]);
    }

    int count = 0;
    for (auto& kv : line_threads) {
        long line = kv.first;
        if (kv.second.size() >= 2 && line_addrs[line].size() >= 2) {
            out[count++] = line;
        }
    }
    return count;
}
