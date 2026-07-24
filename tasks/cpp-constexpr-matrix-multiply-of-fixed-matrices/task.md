## Context

In modern C++, a `constexpr` function can run entirely at **compile time**: if
every value it touches is itself a constant expression, the compiler is
required to be able to evaluate the function during translation and fold the
result into a compile-time constant — no runtime work at all. Matrix
multiplication is a natural example: multiplying two small, fixed-size
matrices whose entries are known at compile time can be done entirely by the
compiler's constant evaluator.

```cpp
template <int R, int C>
struct Mat {
    double d[R * C];   // row-major
};
```

## Task

Implement two `constexpr` functions:

```cpp
constexpr Mat<2, 2> const_matmul_2x3(const Mat<2, 3>& A, const Mat<3, 2>& B);
constexpr Mat<4, 4> const_matmul_4x4(const Mat<4, 4>& A, const Mat<4, 4>& B);
```

Each must compute the row-major matrix product $C = A \times B$:

$$
C_{ij} = \sum_k A_{ik} \, B_{kj}
$$

and — critically — each must be a valid **constant expression**: no dynamic
allocation, no exceptions on the path taken, nothing that the compiler can't
evaluate during translation.

## Example

For $A = \begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$ and
$B = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$:

$$
C = A \times B = \begin{pmatrix}4&5\\10&11\end{pmatrix}
$$

so `const_matmul_2x3` returns `Mat<2,2>{4.0, 5.0, 10.0, 11.0}`.

## What the gate checks

A `constexpr` function's *definition* must be visible in whichever
translation unit evaluates it as a constant expression, so `solve.cpp`
itself contains a compile-time self-check right below where you implement
the two functions (don't edit that part): it builds two problem instances
with integer-valued entries (so double arithmetic stays bit-exact — no
rounding to worry about), assigns your functions' results to `constexpr`
variables, and `static_assert`s every output entry against the true product
computed independently (verified with numpy while authoring this task, then
pinned as literal constants).

If your function can't actually be evaluated as a constant expression, or
computes the wrong values, the `static_assert`s fail and `solve.cpp` fails to
**compile** — which fails the gate exactly like a wrong runtime answer would.
`main.cpp` separately calls your finished functions at runtime on its own
matrices and prints the product, giving the grader numbers to diff once the
file compiles at all.
