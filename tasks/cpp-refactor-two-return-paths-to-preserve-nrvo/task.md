## Context

**Named Return Value Optimization (NRVO)** lets the compiler construct a
named local variable directly in the caller's return-value storage,
eliminating the copy/move constructor call that would otherwise fire when
that local is returned. It's not guaranteed by the standard (unlike
returning a prvalue), but real compilers reliably grant it for the
textbook shape: a **single named local**, declared once at the top of the
function, populated along whichever branch runs, and returned by exactly
**one** `return` statement at the very end:

```cpp
Result make_result(bool cond, ...) {
    Result r;                 // one named local
    if (cond) { r.a = ...; r.b = ...; }
    else      { r.a = ...; r.b = ...; }
    return r;                 // one return, always the same object
}
```

The fixed contract (`sol.hpp`) instruments the return type so every REAL
copy or move constructor call is directly observable:

```cpp
struct Result {
    int a; double b;
    Result(const Result&);   // bumps g_copy_count
    Result(Result&&);        // bumps g_move_count
};
```

## Task

Implement `make_result(bool cond, int a1, double b1, int a2, double b2)`
(declared in `sol.hpp`) in `solve.cpp`. It must return `Result{a1, b1}` when
`cond` is true, `Result{a2, b2}` otherwise — written as a single named local
`Result`, assigned on whichever branch, returned by one `return` statement
at the end (the shape above), so that the compiler can construct it
straight into the caller's storage: zero copies, zero moves.

## Example

For the fixed driver's four scenarios, the correct run prints
(`a`, `b`, `copy_count`, `move_count`):

```
10 3.140000 0 0
20 2.718000 0 0
65 100000.000000 0 0
200 4.560000 0 0
```

A starter that returns a hardcoded, wrong `Result(0, 0.0)` regardless of
`cond` prints `0 0.000000 0 0` for every scenario — wrong on the values
before the copy/move counts even matter.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `max_abs_err <= 1e-9` against the same driver linked
with `ref.cpp`. Both the selected `(a, b)` values AND the `copy_count`/
`move_count` are printed and compared — even a correct-valued
implementation that routes through an extra copy (e.g. building a temporary
somewhere along the way, or taking the value by copy instead of populating
the named local in place) shows up as a nonzero count and fails the gate.
