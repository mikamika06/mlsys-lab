#include <list>
#include <unordered_map>
#include <vector>
#include "sol.hpp"

int count_misses(const uint64_t* addrs, int n,
                  int line_bytes, int sets, int ways) {
    std::vector<std::list<uint64_t>> lists(sets);
    std::vector<std::unordered_map<uint64_t, std::list<uint64_t>::iterator>> pos(sets);

    int misses = 0;
    for (int i = 0; i < n; ++i) {
        uint64_t line = addrs[i] / static_cast<uint64_t>(line_bytes);
        int set_idx = static_cast<int>(line % static_cast<uint64_t>(sets));
        auto& lst = lists[set_idx];
        auto& mp = pos[set_idx];

        auto it = mp.find(line);
        if (it != mp.end()) {
            lst.erase(it->second);
            lst.push_front(line);
            mp[line] = lst.begin();
            continue;  // hit
        }

        ++misses;
        if (static_cast<int>(lst.size()) >= ways) {
            uint64_t victim = lst.back();
            lst.pop_back();
            mp.erase(victim);
        }
        lst.push_front(line);
        mp[line] = lst.begin();
    }
    return misses;
}
