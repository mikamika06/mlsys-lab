## Context

A type should enforce **const-correctness**: when building a view over a
matrix, provide two overloads of element access,

```cpp
double& operator()(long r, long c);              // callable on a mutable view
const double& operator()(long r, long c) const;   // callable on a const view
```

which differ only in the const-ness of `this`. If a `MatrixView` variable
(or reference) is `const`, the compiler statically refuses to call the
non-const overload — a write through a const view is a **compile error**,
not something checked at runtime. This is the same mechanism that lets a
function taking `const MatrixView&` promise its caller "I will not modify
your matrix" and have the compiler enforce that promise.

A `MatrixView` is often a *sub-block* of a larger row-major matrix, not the
whole thing — think of viewing rows 1..3, columns 1..3 of a bigger array.
Then consecutive rows of the view are NOT `cols` elements apart in the
backing storage; they are `row_stride` elements apart, where
`row_stride >= cols`. Element `(r, c)` of the view lives at
`data[r * row_stride + c]`.

## Task

Implement both overloads of `MatrixView::operator()` declared in `sol.hpp`:

```cpp
double& operator()(long r, long c);
const double& operator()(long r, long c) const;
```

Both must compute the same address, `data[r * row_stride + c]` — the
non-const overload returns it as a writable `double&`, the const overload as
a read-only `const double&`.

## Example

For a `MatrixView` over a 4x6 backing array, opened as a 3x3 sub-block
starting at offset 7 with `row_stride = 6`:

```cpp
MatrixView view(buf + 7, 3, 3, 6);
view(0, 0) = 10.0;   // writes buf[7 + 0*6 + 0] = buf[7]
view(1, 2) = 60.0;   // writes buf[7 + 1*6 + 2] = buf[15]
const MatrixView& cview = view;
cview(0, 0);         // reads buf[7] back -> 10.0
```

Using `r * cols + c` instead of `r * row_stride + c` would put `view(1, 0)`
at `buf[10]` instead of `buf[13]` — inside the *next* row's gap, silently
corrupting a cell the view was never supposed to touch.

## What the gate checks

The driver fills a 4x6 backing array with a distinct canary value per cell,
opens a 3x3 view with `row_stride = 6` over a sub-block of it, writes a
fixed pattern through the mutable view, reads it back through a
`const MatrixView&` bound to the same object, and prints both the 9 values
read back and the full 24-cell backing array (so a write that landed on the
wrong cell — clobbering a canary — is visible too). The grader compiles
`solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed value, including every untouched canary, matches the reference}
$$
