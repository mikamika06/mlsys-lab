#include <cstddef>
#include <vector>
#include "sol.hpp"

namespace {
double value(int i, int j) {
    return static_cast<double>((i * 131 + j * 977) % 1009);
}
}  // namespace

double sum_all_columns(int R, int C) {
    int ld = C;
    // Any leading dimension that is a multiple of (cache sets * doubles per
    // line) makes every row alias into the SAME cache set, no matter how
    // many rows or how much total cache capacity there is. Pad it by one
    // cache line's worth of doubles to break that alignment.
    constexpr int kSetsTimesLine = 32 * 8;  // must match the pinned cache geometry (32 sets, 8 doubles/line)
    if (ld % kSetsTimesLine == 0) ld += 8;

    std::vector<double> data(static_cast<size_t>(R) * static_cast<size_t>(ld));
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            data[static_cast<size_t>(i) * ld + j] = value(i, j);

    double sum = 0.0;
    for (int j = 0; j < C; ++j) {
        for (int i = 0; i < R; ++i) {
            touch(&data[static_cast<size_t>(i) * ld + j]);
            sum += data[static_cast<size_t>(i) * ld + j];
        }
    }
    return sum;
}
