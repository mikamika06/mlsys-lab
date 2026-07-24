#include <list>
#include "sol.hpp"

namespace {
struct LevelPWC {
    int cap;
    std::list<int> keys;  // front = most recently used

    bool touch(int key) {
        for (auto it = keys.begin(); it != keys.end(); ++it) {
            if (*it == key) {
                keys.erase(it);
                keys.push_front(key);
                return true;  // hit
            }
        }
        if ((int)keys.size() >= cap) keys.pop_back();
        keys.push_front(key);
        return false;  // miss
    }
};
}  // namespace

long page_walk_cycles(const int* keys, int num_addrs, const int* cap,
                       long hit_cycles, long miss_cycles, long data_cycles) {
    LevelPWC pwc[4] = {{cap[0], {}}, {cap[1], {}}, {cap[2], {}}, {cap[3], {}}};

    long total = 0;
    for (int j = 0; j < num_addrs; j++) {
        for (int i = 0; i < 4; i++) {
            bool hit = pwc[i].touch(keys[j * 4 + i]);
            total += hit ? hit_cycles : miss_cycles;
        }
        total += data_cycles;
    }
    return total;
}
