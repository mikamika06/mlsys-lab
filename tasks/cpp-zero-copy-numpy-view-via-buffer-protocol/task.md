## Context

When Python passes a NumPy array into C++ via pybind11, the data already
lives in a contiguous buffer. A **zero-copy view** reinterprets that buffer
without duplicating any bytes: back in Python, `np.shares_memory(inp, out)`
is `True` and `inp.ctypes.data == out.ctypes.data`, so mutating one array is
immediately visible through the other.

This task models the buffer-protocol side of that in real, self-contained
C++: `ArrayView` (declared in `sol.hpp`) is a minimal stand-in for
pybind11's `buffer_info` / CPython's `Py_buffer` — a pointer, an element
count, an item size, and a byte stride.

## Task

Implement `make_zero_copy_view` in `solve.cpp`:

```cpp
ArrayView make_zero_copy_view(double* arr, long n);
```

Return an `ArrayView` with `buf = arr` (the exact same pointer — do not
allocate a new buffer or copy any elements), `len = n`, and
`itemsize = stride = sizeof(double)`.

The fixed driver in `main.cpp`, for each of three arrays of different
lengths:
1. checks `view.buf == arr` and the other three fields,
2. **writes** through `view.buf`, then reads `arr` directly and prints it —
   a real zero-copy view must show the write.

## Example

```cpp
double arr[3] = {1.0, 2.0, 3.0};
ArrayView v = make_zero_copy_view(arr, 3);
v.buf[0] = 99.0;
// arr[0] is now 99.0 too -- same memory, not a copy.
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across three fixed array
lengths. The starter returns an all-null/zero `ArrayView`, which fails
every printed check — the pointer-equality flags, the field checks, and the
after-write values (since `arr` is never actually touched).
