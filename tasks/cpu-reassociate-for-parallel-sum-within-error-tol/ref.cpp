#include "sol.hpp"
#include <vector>

float parallel_sum(const float* values, int n) {
    std::vector<float> buf(values, values + n);
    std::vector<int> depth(n, 0);
    int m = n;
    while (m > 1) {
        int half = m / 2;
        for (int i = 0; i < half; ++i) {
            buf[i] = buf[2 * i] + buf[2 * i + 1];
            depth[i] = record_add(depth[2 * i], depth[2 * i + 1]);
        }
        if (m % 2 == 1) {
            buf[half] = buf[m - 1];
            depth[half] = depth[m - 1];
            ++half;
        }
        m = half;
    }
    return buf[0];
}
