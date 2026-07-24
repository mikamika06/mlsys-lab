## Context

Under the **Itanium C++ ABI** (used on macOS and Linux, LP64), a class
`Derived` that inherits from multiple bases lays their subobjects out
sequentially in memory. The **first** (primary) base shares `Derived`'s own
vtable-pointer slot at offset 0 — no separate vptr is needed for it. Every
**secondary** base, though, gets placed at whatever offset comes after the
bases before it; if that secondary base is itself polymorphic, it needs its
*own* vptr, and that vptr obviously cannot also live at offset 0 (B1's vptr
is already there).

The consequence: a pointer to a secondary base is **not** the same numeric
address as the `Derived*` it came from. Every time code writes
`static_cast<Base*>(derived_ptr)` for a non-primary base, the compiler
silently emits a **this-pointer adjustment** — it adds a fixed byte offset
(the "thunk delta") to the pointer before handing it to code that expects a
`Base*`. `reinterpret_cast` does **not** do this (it's a bit-pattern
reinterpretation, not a semantically-aware conversion), which is exactly why
`reinterpret_cast` is the wrong tool for casting across bases in a
multiple-inheritance hierarchy.

## Task

Given

```cpp
struct B1 { virtual ~B1(); long x1; };                 // polymorphic, primary base
struct B2 { virtual ~B2(); int x2; double y2; };        // polymorphic, secondary base
struct B3 { char x3; short y3; };                       // non-polymorphic, secondary base
struct Derived : B1, B2, B3 { int extra; };
```

implement

```cpp
void base_offsets(std::size_t offs[4]);
```

filling `offs[0..2]` with the this-pointer adjustment (in bytes) for
`Derived* -> B1*`, `Derived* -> B2*`, `Derived* -> B3*` respectively, and
`offs[3]` with `sizeof(Derived)`.

## Example

`B1` is primary, so `offs[0] = 0`. `B1`'s own subobject occupies
`sizeof(B1)` bytes (its vptr plus `x1`), so `B2`'s subobject — which needs
its own vptr — starts right after that: `offs[1] = sizeof(B1)`. `B3`'s
subobject (no vptr of its own) starts after `B2`'s: `offs[2] = offs[1] +
sizeof(B2)`, possibly rounded up for `B3`'s alignment. `offs[3]` is the
total `sizeof(Derived)`, including `extra` and any trailing padding.

## What the gate checks

The driver calls `base_offsets` and prints all four values. The grader
compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed offset and sizeof matches the reference}
$$

Assuming a pointer cast never changes the pointer's numeric value — true for
casting to the *primary* base, false for every secondary one — gets
`offs[0]` right by coincidence but `offs[1]` and `offs[2]` wrong, since B2
and B3's subobjects genuinely start partway into the object, not at its
front.
