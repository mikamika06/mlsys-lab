# Single-inheritance object layout offsets

## Context

Under the Itanium C++ ABI (the ABI clang uses on macOS/Linux, LP64), a class
with a virtual function is *polymorphic*: every object carries a hidden
**vptr** (an 8-byte pointer to the vtable) placed at offset $0$. Data members
then follow in declaration order, each rounded up to its own alignment, with
padding inserted where needed.

When you derive from a polymorphic base by **single inheritance**, two rules
shape the layout of the derived object:

- The derived class **reuses the base's vptr** — it does *not* add a second
  one. So the vptr stays at offset $0$.
- The derived class's own members are laid out starting from the base's
  **data size** ($\mathrm{dsize}$, the offset just past the base's last member),
  **not** the base's full `sizeof`. This means a derived member can be placed in
  the **tail padding** of the base subobject.

Concretely:

```cpp
struct Base {
    virtual ~Base();   // polymorphic -> vptr
    long  a;           // 8 bytes, align 8
    int   b;           // 4 bytes, align 4
};                     // dsize = 20, sizeof(Base) = 24 (4 tail-pad bytes)

struct Derived : Base {
    char   c;          // 1 byte,  align 1
    short  d;          // 2 bytes, align 2
    int    e;          // 4 bytes, align 4
    double f;          // 8 bytes, align 8
};
```

## Task

Implement, in `solve.cpp`:

```cpp
void derived_layout(std::size_t offs[8]);
```

Fill `offs[]` with the byte offset of each of the following **inside a
`Derived` object**, in exactly this order:

| index | quantity          |
|-------|-------------------|
| 0     | vptr              |
| 1     | `Base::a`         |
| 2     | `Base::b`         |
| 3     | `Derived::c`      |
| 4     | `Derived::d`      |
| 5     | `Derived::e`      |
| 6     | `Derived::f`      |
| 7     | `sizeof(Derived)` |

The `Base` and `Derived` types are fully defined in `sol.hpp`, so you can either
derive the offsets by hand or read them straight from the compiler.

## Example

For the hierarchy above, the driver prints:

```
vptr=0
a=8
b=16
c=20
d=22
e=24
f=32
sizeof=40
```

Note `c=20`: `Base::b` ends at offset $20$ but `sizeof(Base)` is $24$, so
`Derived::c` lands at $20$ — inside the base's tail padding — instead of at $24$.

## What the gate checks

`main.cpp` (fixed) calls your `derived_layout` and prints all eight values. The
grader compiles `main.cpp + solve.cpp` with `clang++ -O2 -std=c++20`, runs it,
and compares the full output to the reference. Metric: **exact_match** — every
printed number must match the reference layout exactly.
