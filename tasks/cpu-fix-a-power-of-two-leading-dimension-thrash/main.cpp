#include <cstdint>
#include <cstdio>
#include <list>
#include <unordered_map>
#include <vector>
#include "sol.hpp"

// FIXED driver + FIXED cache model. 32 sets, 4 ways, 64-byte lines (8 KiB /
// 1024 doubles total) — a real set-associative LRU cache, not a hardware
// counter. Deterministic: no rand()/time(), the matrix contents come from
// the fixed formula the learner's own sweep must also use.

namespace {

constexpr int kLineBytes = 64;
constexpr int kSets = 32;
constexpr int kWays = 4;

struct LruCache {
    std::vector<std::list<uint64_t>> lists;
    std::vector<std::unordered_map<uint64_t, std::list<uint64_t>::iterator>> pos;
    int misses = 0;

    LruCache() : lists(kSets), pos(kSets) {}

    void reset() {
        for (auto& l : lists) l.clear();
        for (auto& p : pos) p.clear();
        misses = 0;
    }

    bool access(uint64_t line) {
        int set_idx = static_cast<int>(line % static_cast<uint64_t>(kSets));
        auto& lst = lists[set_idx];
        auto& mp = pos[set_idx];
        auto it = mp.find(line);
        if (it != mp.end()) {
            lst.erase(it->second);
            lst.push_front(line);
            mp[line] = lst.begin();
            return true;
        }
        ++misses;
        if (static_cast<int>(lst.size()) >= kWays) {
            uint64_t victim = lst.back();
            lst.pop_back();
            mp.erase(victim);
        }
        lst.push_front(line);
        mp[line] = lst.begin();
        return false;
    }
};

LruCache g_cache;

}  // namespace

void cache_reset() { g_cache.reset(); }

bool touch(const void* p) {
    uint64_t line = reinterpret_cast<uint64_t>(p) / static_cast<uint64_t>(kLineBytes);
    return g_cache.access(line);
}

int miss_count() { return g_cache.misses; }

int main() {
    constexpr int R = 64;
    constexpr int C = 256;  // power of two: a multiple of kSets * (kLineBytes / 8)

    cache_reset();
    double sum = sum_all_columns(R, C);
    printf("R=%d C=%d\n", R, C);
    printf("sum=%.1f\n", sum);
    printf("misses=%d\n", miss_count());
    return 0;
}
