## Context

When binding C++ numerical code to Python with tools like **pybind11** or
the raw CPython buffer protocol (`Py_buffer`), the C++ side must validate an
incoming NumPy array before dereferencing its raw data pointer: unlike
dynamically-typed Python, C++ needs a specific element type, an exact number
of dimensions, and (usually) an exact shape. A robust binding checks the
buffer's metadata first and raises the right *kind* of exception — a
`TypeError` for "this isn't even the right sort of object/dtype", a
`ValueError` for "the shape is wrong" — instead of crashing on a bad
pointer.

This task models that validation in real, self-contained C++: `TypeErrorSim`
and `ValueErrorSim` (declared in `sol.hpp`) are ordinary C++ exception types
you `throw` and the driver `catch`es, and `BufferObj` is a minimal stand-in
for a NumPy array's buffer-protocol metadata (dtype string, `ndim`, `shape`,
flattened `data`).

## Task

Implement `validate_buffer` in `solve.cpp`:

```cpp
double validate_buffer(const BufferObj& arr, const std::string& expected_dtype,
                        const int* expected_shape, int expected_ndim);
```

Check, **in this exact order**:

1. `!arr.is_valid_buffer` &rarr; `throw TypeErrorSim(...)`
2. `arr.dtype != expected_dtype` &rarr; `throw TypeErrorSim(...)`
3. `arr.ndim != expected_ndim` &rarr; `throw ValueErrorSim(...)`
4. for $i \in [0, \text{ndim})$: if `expected_shape[i] != -1` and
   `arr.shape[i] != expected_shape[i]` &rarr; `throw ValueErrorSim(...)`

If every check passes, return $\sum_{j=0}^{\text{arr.size}-1} \text{arr.data}[j]$.

## Example

```cpp
// arr: float32, shape (3,3), all 1.0
validate_buffer(arr, "float32", {3, 3}, 2);   // -> 9.0

// wrong dtype
validate_buffer(arr, "float64", {3, 3}, 2);   // throws TypeErrorSim

// wrong shape
validate_buffer(arr, "float32", {3, 4}, 2);   // throws ValueErrorSim

// wildcard first dim: shape (10,5) matches expected (-1,5)
validate_buffer(y, "int32", {-1, 5}, 2);      // -> 50.0
```

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) over 10 fixtures: valid arrays
(including wildcard-shape matches), a non-buffer input, a dtype mismatch, a
rank mismatch, and shape mismatches (including one that fails even with a
wildcard present in another dimension). Each case prints either the summed
value or the exception type name (`TypeError`/`ValueError`) that was raised.
The starter never validates anything and always returns `0.0`, so it is
wrong on every fixture that should raise.
