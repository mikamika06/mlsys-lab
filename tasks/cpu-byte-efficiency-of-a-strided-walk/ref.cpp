#include "sol.hpp"

void walk(int n, int stride_elems, int width) {
    for (int i = 0; i < n; i++) {
        long start = static_cast<long>(i) * stride_elems * width;
        for (int b = 0; b < width; b++) {
            touch(start + b);
        }
    }
}

double byte_efficiency(int n, int stride_elems, int width) {
    reset_touch();
    walk(n, stride_elems, width);
    double bytes_used = static_cast<double>(n) * width;
    double bytes_fetched = static_cast<double>(touched_line_count()) * 64.0;
    return bytes_used / bytes_fetched;
}
