## Context

SFINAE (Substitution Failure Is Not An Error) constrains function templates
with `std::enable_if_t`: if the condition is false, that overload silently
drops out of the candidate set instead of causing an error. It's easy to
write constraints that overlap, though — if two overloads can both be true
for the same type, the compiler has two equally-good candidates and reports
a hard **ambiguous call** error.

```cpp
// Overload 1
template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process(T t);

// Overload 2
template <typename T, std::enable_if_t<(sizeof(T) <= 4), int> = 0>
void process(T t);

// Overload 3
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void process(T t);
```

Under LP64, `int` is 4 bytes and integral — it satisfies both Overload 1 and
Overload 2. `float` is 4 bytes and floating-point — it satisfies both
Overload 2 and Overload 3.

## Task

Fix the `enable_if_t` constraints on the three `process` overloads in your
`.cpp` so every type is routed to **exactly one** overload:

- **Overload 2** should handle ALL types where `sizeof(T) <= 4`.
- **Overload 1** should handle integral types where `sizeof(T) > 4`.
- **Overload 3** should handle floating-point types where `sizeof(T) > 4`.

A type that fits none of these (e.g. an 8-byte pointer) should match no
overload at all.

## Example

With the broken constraints, `int` is ambiguous between Overload 1 and
Overload 2. With the fix, only Overload 2's `sizeof(T) <= 4` holds for
`int` (`sizeof(int) == 4`, and `int` no longer satisfies Overload 1's
`> 4` requirement), so it routes cleanly to Overload 2 (tag `2`).

## What the gate checks

Each `process` overload sets a global tag (1, 2, or 3) when it's the one
selected. The driver tests 9 real types — `bool`, `char`, `short`, `int`,
`long`, `long long`, `float`, `double`, and a pointer — through a small
SFINAE detection idiom that safely asks "does `process(T{})` even compile
for this T?" before calling it, so one bad type can't take down the whole
build. That detection idiom treats both "no matching overload" and
"ambiguous overload" the same way: as `NoMatch`. With the broken
constraints, every small integral type (`bool`, `char`, `short`, `int`) is
ambiguous between Overload 1 and Overload 2, and `float` is ambiguous
between Overload 2 and Overload 3 — so they all silently print `NoMatch`
instead of their intended tag, which is wrong output, not a crash.

The grader compiles your `.cpp` together with the fixed driver using the
real local `clang++`, runs it, and requires the 9 printed lines to match the
reference's exactly ($\mathrm{exact\_match}=1.0$).
