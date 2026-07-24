#pragma once

// Predict sizeof() of a struct whose fields have byte sizes sizes[0..n),
// declared in that order, under the standard C/C++ struct-layout rule for
// fundamental types (natural alignment == size for char/short/int/long/
// double/pointer): each field is placed at the next offset that is a
// multiple of its own size, and the struct's total size is then rounded up
// to a multiple of the LARGEST field size present.
int predict_sizeof(const int* sizes, int n);
