## Context

When writing a C++ extension for Python (e.g. via `pybind11`), you can reach
a NumPy array's underlying memory directly through the **buffer protocol** —
true **zero-copy**: C++ mutates the array's own storage in place, no copy in
or out.

If the array holds structured data (an array of structs), the mutator must
walk it as real, natively-laid-out records — reading/writing the wrong
field means reading/writing the wrong bytes. Three fixed records here
(`sol.hpp`), each mixing a `double` field with other types around it:

```cpp
struct RecordA { char c; double d1; double d2; };
struct RecordB { double d; char c; int i; double d2; };
struct RecordC { int i; double d; float f; double d2; };
```

## Task

Implement, in `solve.cpp`:

- `mutate_a(RecordA* arr, int n)`
- `mutate_b(RecordB* arr, int n)`
- `mutate_c(RecordC* arr, int n)`

Each must add `1.0` to **every** field of type `double` in every one of the
`n` elements, in place — leaving every non-`double` field (the `char`,
`int`, `float` fields) exactly as it was.

## Example

For the fixed driver's seed data, the correct run prints (one line per
element; `RecordA`'s 5 elements, then `RecordB`'s 3, then `RecordC`'s 4):

```
0 2.000000 3.000000
0 3.000000 4.000000
0 4.000000 5.000000
0 5.000000 6.000000
0 6.000000 7.000000
1.000000 1 20 4.000000
2.000000 1 20 5.000000
3.000000 1 20 6.000000
0 2.000000 2.000000 4.000000
0 3.000000 3.000000 5.000000
0 4.000000 4.000000 6.000000
0 5.000000 5.000000 7.000000
```

The `char`/`int`/`float` columns above (`0`, `1 20`, `0`) never change — only
the `double` columns are +1.0 from their seed values. A starter with empty
mutator bodies leaves everything at the seed values (every `double` column
1.0 lower than the reference).

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `max_abs_err <= 1e-9` against the same driver linked
with `ref.cpp`. Touching the wrong field (e.g. bumping `RecordB.i` instead
of `RecordB.d`, or missing one of a struct's two `double` fields) changes a
printed value and fails the gate.
