#include <cstddef>
#include <vector>
#include "sol.hpp"

namespace {
double value(int i, int j) {
    return static_cast<double>((i * 131 + j * 977) % 1009);
}
}  // namespace

double sum_all_columns(int R, int C) {
    int ld = C;  // leading dimension == requested column count, no padding

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
