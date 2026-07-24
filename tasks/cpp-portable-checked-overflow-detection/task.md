## Context

In C++, signed integer overflow (`INT_MAX + 1`, etc.) is **undefined
behavior**. Optimizing compilers exploit that UB by assuming it never
happens — a naive post-operation check like `if (a + b < a)` can get
silently optimized away, because the compiler is allowed to assume `a + b`
never overflows in the first place.

To detect overflow *portably*, without ever triggering it, the trick is to
compute the operation in a type wide enough that it CAN'T overflow there,
then range-check the wide result against the narrow type's real bounds
(`INT_MIN`/`INT_MAX`, `SHRT_MIN`/`SHRT_MAX`, `SCHAR_MIN`/`SCHAR_MAX`,
`LONG_MIN`/`LONG_MAX` — the real compiler's own limits). `__int128` (a
Clang/GCC extension) is wide enough to hold the product of any two 64-bit
values without itself overflowing, so it works uniformly for every
type/operation pair here, including `long op long`.

## Task

Implement, in `solve.cpp`, the twelve functions declared in `sol.hpp` —
`add_overflow_int`, `sub_overflow_short`, `mul_overflow_char`, and so on for
`{add, sub, mul} x {int, short, char (signed char), long}`. Each returns
whether `a OP b` would overflow that type's range, computed **without**
performing the operation directly in the narrow type when it might
overflow.

## Example

```cpp
add_overflow_int(2000000000, 1500000000)   // 3.5B > INT_MAX  -> true
add_overflow_int(-2000000000, 1500000000)  // -500M in range  -> false
mul_overflow_short(300, 300)               // 90000 > SHRT_MAX -> true
```

## Reference numbers

For the fixed driver's twelve scenarios, the correct run prints:

```
1 0 1 1 1 1 1 1 1 0 1 1
```

Notably, scenario 10 (`mul_overflow_long(100000, 100000)`) is `0`
(`false`) — `10^10` comfortably fits in a 64-bit `long`, even though the
same product would overflow a 32-bit `int` (scenario 9, `1`/`true`). A
starter that unconditionally returns `false` prints `0 0 0 0 0 0 0 0 0 0 0 0`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the twelve printed flags against
the same driver linked with `ref.cpp`. Computing the operation directly in
the narrow signed type (inviting real UB that `-O2` is free to exploit), or
getting a boundary wrong (e.g. treating `INT_MIN - 1` as in-range), flips a
flag and fails the gate.
