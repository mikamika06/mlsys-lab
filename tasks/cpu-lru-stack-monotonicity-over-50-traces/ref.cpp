#include <list>
#include "sol.hpp"

long lru_miss_count(const int* ids, int n, int capacity) {
    std::list<int> stack;  // front = most recently used
    long misses = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        auto it = stack.begin();
        for (; it != stack.end(); ++it) {
            if (*it == id) break;
        }
        if (it != stack.end()) {
            stack.erase(it);
            stack.push_front(id);
        } else {
            misses++;
            if ((int)stack.size() >= capacity) stack.pop_back();
            stack.push_front(id);
        }
    }
    return misses;
}
