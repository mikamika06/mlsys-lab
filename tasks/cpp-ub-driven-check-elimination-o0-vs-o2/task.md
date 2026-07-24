## Context

Undefined behavior lets the optimizer reason "this path can never happen"
and delete code accordingly — including safety checks the programmer
clearly intended to run. A classic real-world instance: a check that only
makes sense if some earlier operation had UB.

```cpp
bool check_no_overflow(int x) {
    int y = x + 1;
    if (y < x) return false;   // "if adding 1 wrapped around, y < x"
    return true;
}
```

Signed integer overflow (`x == INT_MAX`, so `x + 1` doesn't fit in an
`int`) is undefined behavior in C++. Because of that, the compiler is
allowed to assume `x + 1` **never** overflows — which means it can prove
`y < x` is always false and delete the whole `if`. At `-O0` a simple build
just executes the arithmetic literally (in practice, wrapping via ordinary
two's-complement hardware addition), so the check still "works" there.
At `-O2` the check is gone: `check_no_overflow(INT_MAX)` returns `true`
(claiming no overflow) even though it plainly did overflow.

The fix is to detect the overflow through a well-defined mechanism instead
of relying on wraparound — e.g. `__builtin_add_overflow`, which computes
the sum into an out-parameter and returns whether it overflowed, with no UB
involved at any optimization level.

## Task

Implement the same overflow-check logic in **both** of these entry points
(declared in `sol.hpp`):

```cpp
__attribute__((optnone)) bool check_no_overflow_noopt(int x);
bool check_no_overflow_opt(int x);
```

`check_no_overflow_noopt` is forced to compile **unoptimized**
(`__attribute__((optnone))`, applied regardless of the file's real `-O2`
build) — this stands in for "what `-O0` does". `check_no_overflow_opt`
compiles normally, at the file's real optimization level — "what `-O2`
does". Both must return `false` exactly when `x == INT_MAX` (adding 1
overflows) and `true` otherwise, **and they must agree with each other on
every input** — the check must survive optimization, not merely happen to
work at one level.

## Example

```cpp
__attribute__((optnone)) bool check_no_overflow_noopt(int x) {
    int y; return !__builtin_add_overflow(x, 1, &y);
}
bool check_no_overflow_opt(int x) {
    int y; return !__builtin_add_overflow(x, 1, &y);
}
// check_no_overflow_noopt(INT_MAX) == false
// check_no_overflow_opt(INT_MAX)   == false   -- agrees, no UB involved
```

## What the gate checks

The driver calls both functions for five fixed values of `x` (including
`INT_MAX`) and prints each pair's results plus whether they agree. The
grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and
requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference. A version that detects overflow via
`y = x + 1; if (y < x) ...` passes for `check_no_overflow_noopt(INT_MAX)`
(unoptimized) but the optimizer legitimately deletes the check inside
`check_no_overflow_opt`, so the two disagree at `x = INT_MAX` and the
printed trace stops matching.
