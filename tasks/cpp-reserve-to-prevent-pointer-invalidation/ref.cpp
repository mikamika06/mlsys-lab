#include <vector>
#include "sol.hpp"

GrowthResult grow_vector(int n_elements, bool reserve_first) {
    std::vector<Item> v;
    if (reserve_first) {
        v.reserve(n_elements);
    }

    int reallocs = 0;
    void* prev = nullptr;
    for (int i = 0; i < n_elements; ++i) {
        v.push_back(Item{i, static_cast<double>(i)});
        void* cur = static_cast<void*>(v.data());
        if (i > 0 && cur != prev) {
            ++reallocs;
        }
        prev = cur;
    }

    bool valid = (n_elements == 0) || (reallocs == 0);
    return GrowthResult{reallocs, static_cast<long>(v.capacity()), valid};
}
