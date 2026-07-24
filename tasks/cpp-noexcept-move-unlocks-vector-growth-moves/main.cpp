#include "sol.hpp"
#include <cstdio>

Counters g_counters;

int main() {
    struct Case { int element_size; int n_pushes; bool noex; };
    static const Case cases[] = {
        {16, 10, true},
        {16, 10, false},
        {8, 5, true},
        {8, 5, false},
        {16, 1, true},
        {8, 17, true},
        {8, 17, false},
    };
    for (const auto& c : cases) {
        GrowthCounts r = simulate_vector_growth(c.element_size, c.n_pushes, c.noex);
        printf("%ld %ld %ld %ld %ld\n", r.copies, r.moves, r.destructions,
               r.total_alloc_bytes, r.final_capacity);
    }
    return 0;
}
