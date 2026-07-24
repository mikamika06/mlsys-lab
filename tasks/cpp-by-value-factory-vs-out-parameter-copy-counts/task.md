## Context

Developers often debate the best way to return a heap-backed object from a
factory function. Two common patterns:

1. **By-value factory**:
```cpp
Matrix make_matrix(int n) {
    return Matrix(n);   // return a prvalue directly
}
Matrix m = make_matrix(n);
```

2. **Out-parameter factory**, chosen by people trying to "avoid the copy":
```cpp
void make_matrix(int n, Matrix& out) {
    Matrix tmp(n);
    out = tmp;           // assign a local into the out-parameter
}
Matrix m;
make_matrix(n, m);
```

Historically, returning large objects by value was avoided because it seemed
to force an extra copy constructor call. But C++17 introduced **guaranteed
copy elision**: for `return Matrix(n);` — a prvalue of the function's own
return type, constructed directly in the return statement — the standard
mandates the object is built straight into the caller's storage. No copy
constructor and no move constructor run, on any compiler, at any
optimization level.

The out-parameter pattern, on the other hand, is usually written by first
building a local (`tmp`) and then assigning it into `out`. Since `tmp` is a
named lvalue, `out = tmp` calls the real copy-assignment operator — a full
deep copy — even though the whole point of the pattern was supposedly to
avoid one.

## Task

Implement both factories in `solve.cpp`:

```cpp
Matrix make_by_value(int n);
void make_out_param(int n, Matrix& out);
```

`Matrix(n)` (already implemented for you — see `sol.hpp`) allocates `n` ints
and fills `data[i] = i * i`.

- `make_by_value`: construct and return the result as a direct prvalue —
  `return Matrix(n);` — so C++17's mandatory elision applies and zero
  copy/move constructors run.
- `make_out_param`: build a local `Matrix tmp(n)`, then `out = tmp;`. This is
  the natural way the out-parameter idiom gets written, and it costs exactly
  one real copy-assignment.

The fixed driver in `main.cpp` resets an instrumented copy counter
(`g_copy_count`, incremented only inside `Matrix`'s copy constructor and
copy-assignment operator — never for moves) immediately before each call,
then prints the resulting matrix contents together with the copy count.

## Example

For `n = 2`: `Matrix(2)` holds `[0, 1]` (`data[i] = i*i`). `make_by_value(2)`
prints that data with `copies=0`. `make_out_param(2, out)` prints the same
data with `copies=1`, because building `tmp` and assigning it into `out`
performs one deep copy.

## What the gate checks

The grader compiles `main.cpp` + your `solve.cpp` with real
`clang++ -O2 -std=c++20`, runs it, and compares stdout byte-for-byte against
the reference build (`exact_match == 1.0`) across several sizes `n`. The
starter returns an empty `Matrix()` from `make_by_value` and leaves `out`
untouched in `make_out_param`, so both the printed contents and the printed
copy counts are wrong for every fixture.
