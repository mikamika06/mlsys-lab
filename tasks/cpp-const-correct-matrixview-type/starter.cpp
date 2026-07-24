#include "sol.hpp"

// TODO: implement both overloads. Element (r, c) of the view lives at
// data[r * row_stride + c] in the backing storage -- NOT r * cols + c,
// since the view may be a sub-block of a larger matrix.

double& MatrixView::operator()(long r, long c) {
    (void)r; (void)c;
    // your code here
    return data[0];
}

const double& MatrixView::operator()(long r, long c) const {
    (void)r; (void)c;
    // your code here
    return data[0];
}
