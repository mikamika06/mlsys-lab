#include "sol.hpp"

// Direct prvalue return: C++17 mandates elision here, so no copy/move ctor runs.
Matrix make_by_value(int n) {
    return Matrix(n);
}

// Idiomatic out-parameter pattern: build a local, then copy-assign it into
// `out`. `tmp` is a named lvalue, so this is a real copy-assignment call.
void make_out_param(int n, Matrix& out) {
    Matrix tmp(n);
    out = tmp;
}
