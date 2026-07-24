#include "sol.hpp"
#include <unordered_map>
#include <set>

long count_invalidations(const WriteEvent* trace, int n) {
    std::unordered_map<long, std::set<int>> line_owners;  // line -> owning cores
    long total = 0;

    for (int k = 0; k < n; k++) {
        long line = trace[k].addr / 64;
        int core = trace[k].core;
        auto& owners = line_owners[line];

        long others = 0;
        for (int c : owners) {
            if (c != core) others++;
        }
        total += others;

        owners.clear();
        owners.insert(core);
    }
    return total;
}
