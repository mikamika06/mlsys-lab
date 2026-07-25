#include "sol.hpp"

void classify_prefetch(const long* const* addrs, const int* lens, int num_patterns, int* out) {
    for (int k = 0; k < num_patterns; k++) {
        const long* a = addrs[k];
        int n = lens[k];
        long stride0 = a[1] - a[0];
        bool constant = (stride0 != 0);
        for (int i = 1; i < n - 1 && constant; i++) {
            if (a[i + 1] - a[i] != stride0) constant = false;
        }
        long mag = stride0 < 0 ? -stride0 : stride0;
        out[k] = (constant && mag < 4096) ? 1 : 0;
    }
}
