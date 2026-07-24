## Context

When a struct or class is part of a shipped ABI (a shared library, a plugin
boundary, anything compiled separately and linked later), the *definition* is
baked into every translation unit that ever `#include`d it. Callers hard-code
the **byte offset** of each field, the total `sizeof` used for allocation and
array striding, the alignment, and — for polymorphic types — the presence and
position of the `vptr`. If you edit the type and those numbers move, old binaries
keep reading the *old* offsets out of the *new* layout: silent corruption, not a
link error.

So "does this edit break the ABI?" is, at the object level, a purely mechanical
question about **memory layout**. Define the edit from an old version to a new
version as **ABI-breaking** iff *any* of the following changes:

$$
\text{break} \iff
\big(\,\text{sizeof} \;\lor\; \text{alignof} \;\lor\; \text{has\_vptr}\,\big)\ \text{changes}
\;\;\lor\;\;
\exists\, f \in \text{common fields}:\ \mathrm{offset}(f)\ \text{changes}
$$

where *common fields* are the fields that exist in both versions (matched by
role, listed in the same order). Some consequences worth internalizing:

- Reordering fields, inserting a field in the middle, or widening a field moves
  offsets and/or `sizeof` → **break**.
- Appending a field at the end grows `sizeof` even though existing offsets are
  untouched → **break** (arrays and allocations shift).
- Retyping a field to a **same-size, same-alignment** type (e.g. `int` →
  `unsigned int`) leaves the bytes exactly where they were → **compatible**.
- Adding the *first* `virtual` makes a class polymorphic (a `vptr` appears at the
  front, shifting everything) → **break**; appending another `virtual` to an
  *already*-polymorphic class only grows the vtable, not the object → **compatible**.
- Slotting a new field into pre-existing **tail padding** so `sizeof` and every
  existing offset stay put → **compatible**.

## Task

Implement

```cpp
int abi_breaks(const Layout& old_v, const Layout& new_v);
```

where `Layout` (declared in `sol.hpp`) is the measured object-layout signature of
one version:

```cpp
struct Layout {
    int size;      // sizeof(T)
    int align;     // alignof(T)
    int vptr;      // 1 if T is polymorphic, else 0
    int nfields;   // number of fields common to both versions
    int off[16];   // off[i] = byte offset of the i-th common field
};
```

Return `1` if the edit `old_v -> new_v` is ABI-breaking under the definition
above, else `0`. The driver measures the real layouts with the compiler
(`sizeof`, `alignof`, `std::is_polymorphic`, and pointer arithmetic for offsets),
so you are classifying **true, compiler-produced numbers** — not guessing.

## Example

Appending a field at the end:

```
old = { size=8,  align=4, vptr=0, nfields=2, off=[0,4]   }   // struct { int a; int b; }
new = { size=12, align=4, vptr=0, nfields=2, off=[0,4]   }   // struct { int a; int b; int c; }
```

Offsets of the common fields `a,b` are unchanged, but `size` went `8 -> 12`, so
`abi_breaks(old, new) == 1`.

Retyping `int -> unsigned int`:

```
old = { size=8, align=4, vptr=0, nfields=2, off=[0,4] }
new = { size=8, align=4, vptr=0, nfields=2, off=[0,4] }
```

Every number matches, so `abi_breaks(old, new) == 0`.

## What the gate checks

The driver runs 12 struct/class edits (append, reorder, retype, widen, add
virtual, pad-fill, ...), calls `abi_breaks` on each, prints the 12-bit verdict
vector plus its decimal encoding. The grader compiles your `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires the printed output to match the
reference **exactly** (`exact_match == 1.0`). Both breaking and compatible cases
are present, so a constant answer (always-break or always-compatible) fails — you
must apply the real layout rule.
