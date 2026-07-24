## Context

When interfacing C++ with Python via pybind11 or the CPython buffer protocol, input NumPy arrays are not always stored contiguously. Slicing (`arr[::2, ::3]`) or transposition (`arr.T`) produce **strided** views where consecutive elements along an axis are separated by non-standard byte offsets, exposed to C++ as a raw pointer plus explicit byte strides `(strideRow, strideCol)`.

For a grid of `struct` elements, element $(i, j)$'s byte address is $i \times \text{strideRow} + j \times \text{strideCol}$, and one particular field within that element lives at a further fixed `fieldOffset` (e.g. `offsetof(Elem, val)`) past that address.

## Task

Implement `stridedRowSums` in `solve.cpp` (declared in `sol.hpp`): for each row `i` in `[0, M)`, sum the `double` field found at

```
buf + i*strideRow + j*strideCol + fieldOffset
```

over every column `j` in `[0, N)`, and write the result into `out[i]`. Read every value with `memcpy` -- these addresses are not guaranteed to be aligned for a direct `double*` dereference, since `strideRow`/`strideCol`/`fieldOffset` can put them anywhere. Do **not** assume `strideRow == N * strideCol` (a sub-sliced or transposed view won't satisfy that).

## Example

```cpp
// struct { char c; double val; }  -> fieldOffset = offsetof(Elem, val) = 8
// M=2, N=2, strideRow=64 (a strided sub-slice of a bigger parent buffer), strideCol=16
// out[0] = value(0,0) + value(0,1)
// out[1] = value(1,0) + value(1,1)
```

## What the gate checks

`main.cpp` builds three real scenarios: a 4x4 stride-2 sub-slice of an 8x8 grid of `struct { char; double; }`, a 3x5 stride-2 sub-slice of a 6x10 grid of `struct { int; float; double; }` (nonzero `fieldOffset`), and a genuinely transposed ("column-major") layout where `strideRow < strideCol`. Your printed row sums are compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Assuming `strideRow == N * strideCol` (i.e. ignoring the passed-in stride and recomputing your own) breaks the sub-sliced and transposed scenarios while silently passing a naive contiguous-only test.
