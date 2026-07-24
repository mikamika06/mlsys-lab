## Context

When a C++ constructor performs multiple heap allocations and a later one
throws, stack unwinding destroys already-constructed members in reverse
order. If every resource is held by an RAII member, its destructor frees the
resource — no leak. Each RAII member is itself an object occupying space in
the class layout: typically a raw pointer, sometimes an ownership flag.

The compiler inserts **inter-field padding** so every field starts at a
multiple of its own alignment (which equals its size for every scalar type
here: `char`/`bool`=1, `short`=2, `int`/`float`=4, `long`/`long long`/
`double`/pointer=8), and appends **tail padding** so the whole class's
`sizeof` is a multiple of its strictest member's alignment. Forgetting tail
padding is a common source of `sizeof`/`offsetof` bugs in binary protocols,
serialization, and placement-new — and it means an array of such objects
would NOT be spaced correctly, corrupting every element after the first.

## Task

Fix `compute_layout` (declared in `sol.hpp`) in `solve.cpp`: given `n` field
types in declaration order, compute each field's offset (into
`out_offsets`) and return the total size, matching what a real C++ compiler
would give a class with exactly those members — inter-field padding AND
tail padding.

The shipped implementation computes inter-field padding correctly but
**never adds tail padding**, so its returned size is too small whenever the
class's strictest alignment doesn't already divide the offset right after
the last field.

## Example

```
["int", "int", "char"]
-> size 12, offsets [0, 4, 8]
   int(4) | int(4) | char(1) | tail(3) = 12   (max align 4; 9 % 4 = 1 -> 3 bytes tail)

["pointer", "char", "pointer", "bool"]
-> size 32, offsets [0, 8, 16, 24]
   ptr(8) | char(1) pad(7) | ptr(8) | bool(1) tail(7) = 32
```

The fixed driver (`main.cpp`) cross-checks your function against 14 field
sequences, comparing it not just to a formula but to `sizeof`/`offsetof` of
**actual, equivalently-declared C++ structs** — the real compiler's own
answer. Two of those cases already land on a multiple of the max alignment
with no tail padding needed (so the bug is invisible there); the rest need
it and expose the bug: e.g. case 1 above, the shipped starter returns `9`
instead of `12`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of every printed computed
size/offset against the same driver linked with `ref.cpp` (which itself
matches the real structs' `sizeof`/`offsetof` on all 14 cases). Missing tail
padding on any case where it's needed changes that case's printed size and
fails the gate.
