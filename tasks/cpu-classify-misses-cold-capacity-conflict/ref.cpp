#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sol.hpp"

namespace {

// Generic set-associative LRU cache, used for BOTH the real cache
// (sets > 1) and the fully-associative reference cache (sets == 1,
// ways == total capacity).
struct LruCache {
    std::vector<std::list<uint64_t>> lists;
    std::vector<std::unordered_map<uint64_t, std::list<uint64_t>::iterator>> pos;
    int ways;

    LruCache(int num_sets, int num_ways) : lists(num_sets), pos(num_sets), ways(num_ways) {}

    // Returns true on hit.
    bool access(uint64_t line, int set_idx) {
        auto& lst = lists[set_idx];
        auto& mp = pos[set_idx];
        auto it = mp.find(line);
        if (it != mp.end()) {
            lst.erase(it->second);
            lst.push_front(line);
            mp[line] = lst.begin();
            return true;
        }
        if (static_cast<int>(lst.size()) >= ways) {
            uint64_t victim = lst.back();
            lst.pop_back();
            mp.erase(victim);
        }
        lst.push_front(line);
        mp[line] = lst.begin();
        return false;
    }
};

}  // namespace

MissCounts classify_misses(const uint64_t* addrs, int n,
                            int line_bytes, int sets, int ways) {
    LruCache real(sets, ways);
    LruCache full(1, sets * ways);
    std::unordered_set<uint64_t> seen;

    MissCounts m{0, 0, 0};
    for (int i = 0; i < n; ++i) {
        uint64_t line = addrs[i] / static_cast<uint64_t>(line_bytes);
        int set_idx = static_cast<int>(line % static_cast<uint64_t>(sets));

        bool real_hit = real.access(line, set_idx);
        bool full_hit = full.access(line, 0);

        if (real_hit) continue;  // hit -- not a miss at all, no category

        bool cold = seen.find(line) == seen.end();
        seen.insert(line);

        if (cold) {
            ++m.cold;
        } else if (!full_hit) {
            ++m.capacity;
        } else {
            ++m.conflict;
        }
    }
    return m;
}
