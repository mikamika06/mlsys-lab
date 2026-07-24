#include "sol.hpp"

int predict_sizeof(const int* sizes, int n) {
    int offset = 0;
    int max_align = 1;
    for (int i = 0; i < n; i++) {
        int s = sizes[i];
        if (s > max_align) max_align = s;
        int rem = offset % s;
        if (rem != 0) offset += s - rem;
        offset += s;
    }
    int rem = offset % max_align;
    if (rem != 0) offset += max_align - rem;
    return offset;
}
