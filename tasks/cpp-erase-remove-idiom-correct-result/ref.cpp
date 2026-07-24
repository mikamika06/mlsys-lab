#include "sol.hpp"
#include <algorithm>

void eraseByKey(std::vector<Record>& v, int target) {
    v.erase(std::remove_if(v.begin(), v.end(),
                            [target](const Record& r) { return r.key == target; }),
            v.end());
}
