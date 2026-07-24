#include "sol.hpp"
#include <algorithm>

int minimal_sizeof(const int* sizes, int n) {
    int buf[64];
    for (int i = 0; i < n; i++) buf[i] = sizes[i];
    std::sort(buf, buf + n, [](int a, int b) { return a > b; });

    int offset = 0;
    int max_align = 1;
    for (int i = 0; i < n; i++) {
        int s = buf[i];
        if (s > max_align) max_align = s;
        int rem = offset % s;
        if (rem != 0) offset += s - rem;
        offset += s;
    }
    int rem = offset % max_align;
    if (rem != 0) offset += max_align - rem;
    return offset;
}
