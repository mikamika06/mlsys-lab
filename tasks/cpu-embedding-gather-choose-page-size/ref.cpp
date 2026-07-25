#include "sol.hpp"

long choose_page_size(const int* indices, int n, int row_bytes, const long* page_sizes, int p) {
    long best = -1;
    int best_misses = -1;
    for (int k = 0; k < p; k++) {
        long pb = page_sizes[k];
        tlb_reset(pb);
        for (int i = 0; i < n; i++) {
            long addr = static_cast<long>(indices[i]) * row_bytes;
            touch_addr(addr);
        }
        int m = tlb_miss_count();
        if (best == -1 || m < best_misses || (m == best_misses && pb < best)) {
            best = pb;
            best_misses = m;
        }
    }
    return best;
}
