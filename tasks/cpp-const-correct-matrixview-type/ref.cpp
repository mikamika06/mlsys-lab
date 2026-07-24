#include "sol.hpp"

double& MatrixView::operator()(long r, long c) {
    return data[r * row_stride + c];
}

const double& MatrixView::operator()(long r, long c) const {
    return data[r * row_stride + c];
}
