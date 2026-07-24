#pragma once
#include <stdexcept>
#include <string>

// Real C++ analogues of the two pybind11/buffer-protocol exception types
// this task cares about — ordinary C++ exceptions, thrown with `throw` and
// caught with `catch`. (Defined inline here so both solve.cpp and main.cpp
// see the exact same types.)
struct TypeErrorSim : std::runtime_error {
    explicit TypeErrorSim(const std::string& msg) : std::runtime_error(msg) {}
};
struct ValueErrorSim : std::runtime_error {
    explicit ValueErrorSim(const std::string& msg) : std::runtime_error(msg) {}
};

// A minimal stand-in for a NumPy array's buffer-protocol metadata (harness
// code, built by main.cpp).
struct BufferObj {
    bool is_valid_buffer;   // false models "not actually an ndarray"
    std::string dtype;      // e.g. "float32"
    int ndim;
    int shape[3];            // shape[0 .. ndim)
    double data[64];         // flattened elements, row-major
    int size;                 // total element count (product of shape)
};

// Validate `arr` against expected_dtype / expected_shape (expected_shape
// has expected_ndim entries; -1 is a wildcard matching any size >= 1),
// checking IN ORDER:
//
//   1. !arr.is_valid_buffer                                   -> throw TypeErrorSim
//   2. arr.dtype != expected_dtype                              -> throw TypeErrorSim
//   3. arr.ndim != expected_ndim                                -> throw ValueErrorSim
//   4. for i in [0, ndim): expected_shape[i] != -1 &&
//                           arr.shape[i] != expected_shape[i]    -> throw ValueErrorSim
//
// If all checks pass, return the sum of arr.data[0 .. arr.size).
double validate_buffer(const BufferObj& arr, const std::string& expected_dtype,
                        const int* expected_shape, int expected_ndim);
