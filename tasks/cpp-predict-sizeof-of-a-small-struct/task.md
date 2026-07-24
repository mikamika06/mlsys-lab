## Context

A C++ compiler lays out a struct's members in declaration order, inserting
**inter-field padding** so each member begins at an address that is a
multiple of its own natural alignment. After the last member, it inserts
**tail padding** so the struct's total size is a multiple of its strictest
member's alignment (so an array of the struct keeps every element aligned).

For the fundamental types used here, natural alignment equals size:

| Type | Size (bytes) |
|---|---|
| `char` | 1 |
| `short` | 2 |
| `int` | 4 |
| `long`, `double`, pointer | 8 |

## Task

Implement

```cpp
int predict_sizeof(const int* sizes, int n);
```

which, given the field byte sizes `sizes[0..n)` **in declaration order**,
returns what `sizeof` the corresponding struct would be:

1. Walk the fields; place field `i` at the next offset that is a multiple of
   `sizes[i]`, inserting padding before it if needed.
2. After the last field, round the total size up to a multiple of the
   **largest** field size seen (the struct's own alignment).

## Example

`predict_sizeof({1, 4, 8}, 3)` (a `char`, then an `int`, then a `double`):
the `char` sits at offset 0 (1 byte); the `int` needs 4-byte alignment, so 3
bytes of padding are inserted before it, placing it at offset 4 (4 bytes,
ending at 8); the `double` needs 8-byte alignment, already satisfied, so it
sits at offset 8 (8 bytes, ending at 16). Total 16 — already a multiple of
the struct's 8-byte alignment, so no tail padding is needed. Result: `16`.

## What the gate checks

The driver runs `predict_sizeof` on the field-size lists of eight real
2-3-field C++ structs (never shown to your function directly — only their
field sizes are) and prints your prediction next to that struct's *real*
`sizeof`, taken straight from the compiler. The grader compiles `solve.cpp`
with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the eight predictions matches the reference}
$$

Assuming `sizeof` is just the sum of the field sizes (ignoring padding)
gets some cases right by luck — whenever fields already happen to be in
non-decreasing size order — but is wrong on any struct where a smaller field
precedes a larger one.
