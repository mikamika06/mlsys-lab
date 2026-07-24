#include "sol.hpp"

void stream_copy(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) {
        store_nontemporal(&dst[i], src[i]);
    }
}
