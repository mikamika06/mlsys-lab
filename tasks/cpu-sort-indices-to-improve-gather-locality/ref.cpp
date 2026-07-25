#include "sol.hpp"
#include <algorithm>
#include <vector>

void gather_sorted(const float* table, int table_len, const int* indices,
                    int n, float* output) {
    (void)table_len;
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return indices[a] < indices[b];
    });

    for (int oi : order) {
        int idx = indices[oi];
        touch(table_addr(idx));
        output[oi] = table[idx];
    }
}
