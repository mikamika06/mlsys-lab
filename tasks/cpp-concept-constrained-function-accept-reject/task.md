## Context

A constrained function template can participate in overload resolution only when its constraints are satisfied. In C++20, a concept is a compile-time predicate over template arguments, checked during substitution: if a requirement is not satisfied, the candidate is silently removed instead of producing a hard compilation error — the modern replacement for hand-rolled SFINAE.

Consider a concept that accepts a type only when the expression $x + x$ is valid and its result type is exactly $T$ again:

$$
\mathrm{Acceptable}(T) \iff \texttt{requires(T x) \{ \{x+x\} -> std::same\_as<T>; \}}.
$$

Note that this is a genuine compile-time check against **this platform's real compiler**, not an approximation: integer promotion means `bool + bool` and `char + char` actually produce `int`, not `bool`/`char`, so those two built-ins reject the constraint even though they support `+` at all.

## Task

`sol.hpp` defines six probe types with different `operator+` shapes (given, fixed). Implement `void classify_accepts(int out[12])`. Inside it, define the concept

```cpp
template<class T>
concept Acceptable = requires(T x) {
    { x + x } -> std::same_as<T>;
};
```

and fill `out[0..11]` with `1` when `Acceptable<T>` holds for the probe type at that position and `0` otherwise, testing these twelve types in exactly this order:

0. `int`
1. `double`
2. `float`
3. `bool`
4. `long`
5. `char`
6. `SelfAdd` — `operator+` returns `SelfAdd`
7. `DifferentReturn` — `operator+` returns `long`, not `DifferentReturn`
8. `MissingAdd` — no `operator+` at all
9. `AmbiguousAdd` — a member and a free `operator+` make `x + x` ambiguous
10. `DeletedAdd` — `operator+` is `= delete`d
11. `MixedOnly` — `operator+` only exists for `(MixedOnly, int)`, not `(MixedOnly, MixedOnly)`

## Example

`Acceptable<int>` is `true` (`int + int` is `int`), so `out[0] = 1`. `Acceptable<MissingAdd>` is `false` (no `operator+` exists at all, so the requirement is never satisfied), so `out[8] = 0`.

## What the gate checks

`main.cpp` calls `classify_accepts` and prints each of the twelve entries on its own line. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's stdout, which is produced by a genuine `requires`-expression compiled by the real compiler — not a hardcoded table. Guessing the "obvious" answer (e.g. assuming every arithmetic-looking built-in accepts, or that a deleted operator still counts as present) does not survive contact with actual overload resolution.
