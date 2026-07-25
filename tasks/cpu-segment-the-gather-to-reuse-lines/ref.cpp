#include <algorithm>
#include <vector>
#include "sol.hpp"

void segmented_gather(const float* data, int dsize, const int* idx, int n, float* out) {
    (void)dsize;
    std::vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    // Segment/sort the request stream by target index so repeats of the
    // same index land back-to-back and reuse the line instead of
    // re-fetching it.
    std::sort(order.begin(), order.end(), [&](int a, int b) { return idx[a] < idx[b]; });

    for (int k = 0; k < n; k++) {
        int i = order[k];
        int id = idx[i];
        touch((long)id * (long)sizeof(float));
        out[i] = data[id];
    }
}
