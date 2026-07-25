#include "sol.hpp"
#include <vector>

void gather_dedup(const float* table, int table_len, const int* indices,
                   int n, float* output) {
    std::vector<bool> seen(table_len, false);
    std::vector<float> value_cache(table_len, 0.0f);
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        if (!seen[idx]) {
            seen[idx] = true;
            touch(table_addr(idx));
            value_cache[idx] = table[idx];
        }
        output[i] = value_cache[idx];
    }
}
