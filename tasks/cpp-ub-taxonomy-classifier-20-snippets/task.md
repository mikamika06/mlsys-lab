## Context

Undefined behavior (UB) is code the C++ standard imposes no requirements on. A
compiler may assume UB never happens and optimize accordingly, so recognizing it
is a core skill. This task exercises six common categories:

- **Signed overflow** — signed integer arithmetic whose true result does not fit
  in the type is UB. Unsigned arithmetic instead wraps modulo $2^{w}$ and is
  always defined.
- **Out-of-bounds read** — accessing an array element outside $[0,\text{len})$.
- **Uninitialized read** — reading an automatic (non-`char`) object before it is
  given a value.
- **Null dereference** — dereferencing a null pointer.
- **Oversized / negative shift** — for `E1 << E2`, C++20 makes the behavior
  undefined iff $E2 < 0$ or $E2 \ge w$, where $w$ is the width of the promoted
  left operand. Note that in C++20 a *negative left operand* (e.g. `-1 << 1`) is
  **well-defined** (the result is taken modulo $2^{w}$).
- **Strict aliasing** — reading an object through a pointer/reference of an
  incompatible type (via `reinterpret_cast`) is UB; copying the bytes with
  `std::memcpy` (or accessing through a `char*`) is well-defined.

Each snippet is handed to you as a `Snippet` (see `sol.hpp`). The numeric fields
are interpreted by category (`op`, a `Category`):

| `op`           | meaning of the fields |
|----------------|-----------------------|
| `SIGNED_ADD`   | signed `a + b` in a `width`-bit type |
| `UNSIGNED_ADD` | unsigned `a + b` in a `width`-bit type |
| `ARRAY_IDX`    | access index `b` of an array of length `a` |
| `UNINIT_READ`  | `flag` = 1 initialized before read, 0 uninitialized |
| `NULL_DEREF`   | `flag` = 1 pointer is null, 0 pointer is valid |
| `SHIFT`        | shift a `width`-bit value left by `b` positions |
| `TYPE_PUN`     | `flag` = 0 `reinterpret_cast`, 1 `memcpy` |

## Task

Implement in `solve.cpp`:

```cpp
int classify_ub(const Snippet& s);
```

Return `1` if the snippet is undefined behavior under the C++20 rules above, or
`0` if it is well-defined. Dispatch on `s.op` and apply the rule for that
category. The driver `main.cpp` evaluates a fixed array of 20 snippets, prints
the resulting bit vector, then a packed integer and the UB count.

## Example

```cpp
Snippet a{ SIGNED_ADD, 2147483647LL, 1LL, 32, 0 }; // INT_MAX + 1  -> 1 (overflow)
Snippet b{ SHIFT,      -1LL,         1LL, 32, 0 };  // -1 << 1      -> 0 (defined in C++20)
Snippet c{ ARRAY_IDX,  10LL,         10LL, 32, 0 }; // a[10], len 10 -> 1 (OOB)
```

For the fixed 20 snippets the reference bit vector is:

```
1 0 0 0 1 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1
```

(10 of the 20 snippets are UB.)

## What the gate checks

The grader compiles `main.cpp` with your `solve.cpp` using
`clang++ -O2 -std=c++20`, runs it, and compares the printed output against the
reference implementation. The gate is `exact_match == 1.0`: your 20-bit vector
(and the derived packed value and count) must match the reference exactly. The
starter classifies everything as well-defined, so it fails until you implement
the real rules.
