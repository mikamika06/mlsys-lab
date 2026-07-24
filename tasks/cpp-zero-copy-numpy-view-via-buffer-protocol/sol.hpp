#pragma once
#include <cstddef>

// A minimal stand-in for pybind11::buffer_info / Py_buffer: describes a 1D
// array view as a pointer, an element count, an item size, and a byte
// stride between elements.
struct ArrayView {
    double* buf;
    long len;         // number of elements
    long itemsize;    // bytes per element
    long stride;      // byte stride between elements
};

// Build a ZERO-COPY view of `arr` (an existing contiguous buffer of `n`
// doubles): the returned ArrayView must ALIAS arr's own memory
// (`buf == arr`, exactly the same pointer), not a copy.
//
// Do NOT allocate a new buffer and copy elements into it — that defeats
// the entire point of a zero-copy buffer-protocol view: `np.shares_memory`
// must see the same underlying pointer, and writes through the returned
// view must be visible through `arr` too.
ArrayView make_zero_copy_view(double* arr, long n);
