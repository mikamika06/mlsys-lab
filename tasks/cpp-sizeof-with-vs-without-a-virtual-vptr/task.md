## Context

Under the Itanium C++ ABI (Clang/GCC on LP64 platforms), a class with at
least one `virtual` member stores a hidden **vptr** — a pointer to the
class's vtable — at the beginning of every object. This pointer is 8
bytes with 8-byte alignment, identical in layout terms to a `long` or a
raw pointer.

Giving a class a virtual function is therefore, for layout purposes,
equivalent to prepending a hidden pointer-typed field before all the
user-declared fields: every field's offset shifts forward by 8 bytes, and
the struct's overall alignment becomes at least 8.

## Task

Implement, in `solve.cpp`,

```cpp
long virtual_sizeof(long plain_size, long plain_align);
```

Given the REAL, compiler-computed `sizeof` (`plain_size`) and `alignof`
(`plain_align`) of a plain, non-virtual aggregate, compute the `sizeof`
the same fields would have if the class also had a virtual function:

1. The vptr occupies the first 8 bytes; every user field's offset shifts
   forward by exactly 8. This is always valid here because 8 is a
   multiple of every individual field's alignment in this ABI's
   primitive set `{1, 2, 4, 8}` — shifting the whole block by 8 leaves
   every field's internal padding exactly as it was.
2. The struct's overall alignment becomes `max(plain_align, 8)`.
3. The final size is padded up to a multiple of that new alignment.

## Example

For `struct { int a; char b; }`: `plain_size = 8`, `plain_align = 4`
(`int@0` (4 bytes) + `char@4` (1 byte) + 3 bytes tail padding). Adding a
vptr: alignment becomes `max(4, 8) = 8`; size becomes `8 + 8 = 16`,
already a multiple of 8, so `virtual_sizeof(8, 4) == 16`.

## What the gate checks

The fixed driver (`main.cpp`) declares five real plain aggregates and
their real polymorphic twins (identical fields, plus a virtual
destructor) — both compiled for real, so `sizeof` of the polymorphic
versions is genuine compiler output, not a hand-typed answer. For each
pair it prints the plain `sizeof`/`alignof`, your `virtual_sizeof`
result, and the actual compiled polymorphic `sizeof`. The gate is an
exact string match (`exact_match == 1.0`) against the reference's printed
output — the reference's `computed` value matches the real `actual`
value on every line, so a wrong formula shows up immediately as your
`computed` diverging from what the real compiler produces.
