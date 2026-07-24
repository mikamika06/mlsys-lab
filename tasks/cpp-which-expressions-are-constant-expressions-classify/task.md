## Context

In C++ a *core constant expression* is one the compiler can fully evaluate at
translation time. Only such expressions may appear where the language demands a
compile-time value: an array bound `int a[E];`, a non-type template argument
`std::array<int, E>`, a `static_assert(E)` operand, or the initializer of a
`constexpr` variable.

Whether an expression qualifies depends on *what it reads and what it calls*,
not on its syntactic shape. The rules that matter here:

- A `constexpr` variable is always a constant expression.
- A `const` integral variable is a constant expression **only if** its own
  initializer was a constant expression. A `const int` initialized from a
  runtime value is not.
- Reading a non-`const` (mutable) object is never a constant expression.
- A call `f(args...)` is a constant expression only if `f` is `constexpr`
  **and** every argument is itself a constant expression.
- Indexing `arr[i]` is a constant expression only if `arr` is a constant array
  **and** `i` is a constant expression.
- `sizeof(...)` is always a constant expression; its operand is unevaluated.

Assume these declarations are in scope:

```cpp
constexpr int A = 7;                 // constexpr int
const int     B = 3;                 // const int, constant initializer
int           D = 5;                 // mutable global (runtime value)
const int     C = D + 1;             // const int, initialized from a runtime value
constexpr int arr[4] = {2, 4, 6, 8}; // constexpr array
constexpr int sq(int x){ return x*x; }  // constexpr function
int           rt(int x){ return x+D; }  // ordinary (non-constexpr) function
```

## Task

Classify each of the following 12 expressions as usable in a
constant-expression context (**yes**) or not (**no**):

| # | expression        | # | expression          |
|---|-------------------|---|---------------------|
| 1 | `A * 2`           | 7 | `rt(3)`             |
| 2 | `B + 1`           | 8 | `arr[3]`            |
| 3 | `C - 1`           | 9 | `arr[D % 4]`        |
| 4 | `D + 0`           |10 | `sizeof(arr)`       |
| 5 | `sq(A)`           |11 | `A > B ? A : B`     |
| 6 | `sq(D)`           |12 | `A + D`             |

Implement `unsigned classify_constexpr()` in `solve.cpp` so it returns a 12-bit
mask: set bit `i` (for `i = 0..11`) to `1` if and only if expression number
`i + 1` is a constant expression. Bits 12..31 must stay `0`. You may hand-encode
your reasoning as a literal, OR-in each bit, or build a compile-time detector.

## Example

If only expressions 1 and 8 were constant expressions, the correct mask would be
`(1u << 0) | (1u << 7)` = `129`, and the driver would print:

```
1 0 0 0 0 0 0 1 0 0 0 0
mask=129 popcount=2
```

## What the gate checks

`main.cpp` calls `classify_constexpr()` and prints the 12 bits, the 12-bit mask
value, and the popcount. The grader compiles `main.cpp` with your `solve.cpp`
using `clang++ -O2 -std=c++20` and compares the full printed output to the
reference, which decides every bit from clang++'s own constant-expression
evaluation. Metric: `exact_match == 1.0` — all 12 classifications must be
correct.
