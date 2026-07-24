## Context

In modern C++, returning a local variable by value (`return w;`) makes the
function eligible for **Named Return Value Optimization (NRVO)**: the
compiler constructs the object directly in the caller's return slot, with
zero copies and zero moves.

Developers sometimes write `return std::move(w);`, assuming it guarantees
an efficient move. It does the opposite: `std::move(w)` casts the local
variable to an rvalue reference, so the return expression is no longer a
plain local-variable identifier. That single fact **blocks NRVO** — the
compiler can no longer construct the result in place, and is forced to
move-construct it instead (`move_count` becomes `1` where it could have
been `0`). `return std::move(local);` is a real, well-known pessimization.

## Task

`solve.cpp` ships `make_widget` written with this bug. Fix it so
`Widget make_widget(int v)` returns its local `Widget` in a way that
restores NRVO — zero copies, zero moves — instead of forcing a move.

```cpp
Widget make_widget(int v);
```

The fix is one word: return the plain local variable, not
`std::move(the local variable)`.

## Example

Before the fix, `make_widget(42)` move-constructs the result
(`move_count == 1`). After removing `std::move`, the exact same call
constructs the result in place with `move_count == 0` and
`copy_count == 0` — `w.value` is unchanged either way, only the
construction cost changes.

## What the gate checks

The fixed driver (`main.cpp`) calls `make_widget` for two fixed values,
resetting the real copy/move counters between calls, and prints each
result's value plus the observed `copy_count`/`move_count`. The gate is
an exact string match (`exact_match == 1.0`) against the reference's
printed output: leaving the `std::move` in place keeps `move_count` at
`1` instead of `0` and fails the gate.
