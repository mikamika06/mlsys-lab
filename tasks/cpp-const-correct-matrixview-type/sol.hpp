#pragma once

// A non-owning view over a rows x cols block of doubles that may be a
// sub-block of a larger row-major matrix: row r, column c of the VIEW lives
// at data[r * row_stride + c] in the backing storage (row_stride >= cols;
// row_stride == cols only when the view covers a whole matrix with no gap
// between its rows).
//
// Const-correctness: operator() called on a plain `MatrixView&` must return
// a WRITABLE `double&`; operator() called on a `const MatrixView&` must
// return a READ-ONLY `const double&`. Because the two overloads differ only
// in the const-ness of `this`, the compiler picks the right one
// automatically -- and, used correctly elsewhere, refuses at COMPILE time
// to let a write happen through a const view.
struct MatrixView {
    double* data;
    long rows;
    long cols;
    long row_stride;

    MatrixView(double* d, long r, long c, long stride)
        : data(d), rows(r), cols(c), row_stride(stride) {}

    // Writable access (only callable on a non-const MatrixView).
    double& operator()(long r, long c);

    // Read-only access (the only overload callable on a const MatrixView).
    const double& operator()(long r, long c) const;
};
