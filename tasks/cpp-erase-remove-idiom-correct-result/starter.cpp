#include "sol.hpp"

// BUG: erases while iterating with a plain index that always advances. When
// an element is erased, the next element shifts down into the same slot --
// but `i` still moves forward on the next loop step, so that shifted-down
// element is never checked. Consecutive matches only get every OTHER one
// removed.
void eraseByKey(std::vector<Record>& v, int target) {
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i].key == target) {
            v.erase(v.begin() + i);
        }
    }
}
