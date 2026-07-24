## Context

C++ lays out struct fields sequentially in declaration order. To satisfy
each field's natural alignment (under LP64: `char`=1, `short`=2, `int`=4,
`long`/`double`/pointer=8), the compiler inserts padding before any field
that would otherwise start at a non-aligned offset, plus tail padding so the
whole struct's size is a multiple of its largest field's alignment.

Interleaving small fields (like `char`) between large ones (like `double`)
wastes a lot of space to padding. **Sorting fields in descending order of
size/alignment eliminates all of that internal padding** — placing the
largest-aligned field first means every subsequent, smaller field can be
packed immediately after the previous one with no gap, until the natural
tail-rounding at the very end. This is the standard, provably-optimal
strategy whenever every field's alignment equals its own size (true of all
the fundamental types above).

## Task

Implement

```cpp
int minimal_sizeof(const int* sizes, int n);
```

Given the byte sizes of `n` fields (in some arbitrary, possibly
padding-heavy order), compute the smallest `sizeof()` achievable by
**freely reordering those exact same fields** into a single struct:

1. Sort the sizes in descending order.
2. Lay them out: each field goes at the next offset that is a multiple of
   its own size (inserting padding if needed).
3. Round the final total size up to a multiple of the largest field size.

## Example

```
sizes = {1, 4, 1, 8}   // char, int, char, double, badly ordered
sorted descending: {8, 4, 1, 1}
offset 0: place the 8-byte field  -> offset 8
offset 8: place the 4-byte field  -> offset 12
offset 12: place a 1-byte field   -> offset 13
offset 13: place a 1-byte field   -> offset 14
round up to a multiple of 8       -> 16
minimal_sizeof({1,4,1,8}, 4) == 16
```

## What the gate checks

The driver defines, for four field-size cases, both a real "naive" struct
(the fields in a deliberately bad order) and a real "opt" struct (the fields
sorted descending, the provably minimal arrangement) — both real C++ types,
so `sizeof()` on them is genuine compiler-computed truth. It prints the
naive size, your `minimal_sizeof` prediction, and the real optimal size for
each case. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference — your predicted minimum must equal the real
compiler's `sizeof()` for the best possible field ordering, for every case.
