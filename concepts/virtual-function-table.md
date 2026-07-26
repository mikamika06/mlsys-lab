---
title: "What is virtual function table?"
description: "Virtual function table explained, with a measured sizeof-vs-vptr table from real clang++ output you can reproduce, plus a graded C++ exercise."
datePublished: 2026-07-26
dateModified: 2026-07-26
author: Oleksandr Savkov
---

# What is virtual function table?

A virtual function table is a per-class array of function pointers, built once by the
compiler, that a call through a base pointer reads to find the correct override at runtime.
Every object of that class pays for it with one hidden pointer, and that pointer costs exactly
8 bytes whether the class declares 1 virtual method or 40. Below, six real structs compiled by
clang++ show where those bytes land, byte by byte.

## Virtual table, vtable, vptr — one idea, three names

"Virtual table" and "vtable" are the short names people say out loud; "virtual function table"
is the fuller term compiler and ABI documentation use for the identical structure — one array of
function pointers per polymorphic class, reached from every object through a hidden pointer,
the **vptr**. No separate concept hides behind any of the three names. The rest of this page
says "vtable" and "vptr" because that is what the C++ ABI itself calls them.

## How it works

A class becomes *polymorphic* the moment it declares (or inherits) a virtual function. The
compiler gives the class exactly one vtable — a static array holding one function pointer per
virtual method, in declaration order — and inserts one hidden pointer field, the vptr, into
every object of that class. The vptr is not a per-method cost: it is set once, in the
constructor, and points at the whole table regardless of whether the table holds one entry or
forty. That is the fact people most often get wrong — "more virtual methods" sounds like it
should mean more pointers per object, but it means a longer table shared by every instance.

A call written as `base_ptr->method()` compiles to: load the vptr, load the target function
pointer from a fixed slot in the table, call through it. That is two dependent memory reads
before the real work starts, which is why a call the compiler can prove the target of gets
**devirtualized** into a direct call with zero of those reads — the same kind of "what can the
compiler see statically" question that decides whether [loop
unrolling](loop-unrolling.md) removes a loop's control overhead entirely or only shrinks it.
Where the target cannot be known, the pointer chase is the runtime cost of the flexibility, the
same trade a [warp divergence](warp-divergence.md) pays for letting lanes pick their own control
flow instead of fixing it at compile time.

Two comparisons place the vptr for what it structurally is. Python's descriptor protocol also
resolves a name to different code at lookup time, so [python
descriptors](python-descriptors.md) are the dynamic-language cousin of dispatch deferred to the
moment of use. And `__slots__` replaced a per-instance `__dict__` with a fixed class-level
layout — the same "push shared structure up to the class, keep only a pointer in the instance"
move a vtable makes — which is why [python slots](python-slots.md) is the closest Python-side
analogy. Chasing the vptr is also a memory access whose cost depends on where the table sits in
cache, the same "an unpredictable access is the expensive one" idea behind [cache
locality](cache-locality.md) and [memory coalescing](memory-coalescing.md). What none of those
neighbors share with a vtable is inheritance geometry: under multiple or virtual bases, casting
a `Derived*` to a non-primary `Base*` can require adjusting the pointer's value, because that
base's subobject does not start at offset 0.

## sizeof measured against virtual-method count and inheritance shape

Six structs were compiled for real with clang++: an empty struct, one with a single `int`, the
same struct given one virtual method, the same again given four virtual methods, a struct that
inherits its single virtual method through `virtual` inheritance, and a two-level `Derived :
Base` hierarchy. `sizeof` and, for `Derived`, every member's byte offset were read from the
compiled binary's own output — not computed by hand.

| struct | fields | virtual methods | sizeof (bytes) |
|---|---|---|---|
| `Empty` | 0 | 0 | 1 |
| `OneInt` | 1 `int` | 0 | 4 |
| `OneVirtual` | 1 `int` | 1 | 16 |
| `ManyVirtual` | 1 `int` | 4 | 16 |
| `VDerived : virtual VBase` | 1 `int` | 1 (inherited) | 16 |
| `Derived : Base` | `long`+`int` (Base) + `int` (Derived) | 1 (inherited) | 24 |

| offset in `Derived` | contents |
|---|---|
| 0 | vptr |
| 8 | `Base::a` (`long`) |
| 16 | `Base::b` (`int`) |
| 20 | `Derived::c` (`int`) |
| 24 | `sizeof(Derived)` |

Reproduce it — a real `.cpp` file, compiled by the local `clang++` and run as a subprocess, so
every number is the compiler's, not a claim:

```bash
pip install mlsys-lab
python3 - <<'PY'
import subprocess, tempfile, os

SRC = r"""
#include <cstdio>
#include <cstddef>

struct Empty {};
struct OneInt { int x; };
struct OneVirtual { int x; virtual void f() {} };
struct ManyVirtual { int x; virtual void f() {} virtual void g() {} virtual void h() {} virtual void k() {} };
struct VBase { virtual void f() {} };
struct VDerived : virtual VBase { int y; };
struct Base { virtual void f() {} long a; int b; };
struct Derived : Base { int c; };

int main() {
    printf("Empty sizeof=%zu\n", sizeof(Empty));
    printf("OneInt sizeof=%zu\n", sizeof(OneInt));
    printf("OneVirtual sizeof=%zu\n", sizeof(OneVirtual));
    printf("ManyVirtual sizeof=%zu\n", sizeof(ManyVirtual));
    printf("VDerived sizeof=%zu\n", sizeof(VDerived));
    printf("Derived sizeof=%zu\n", sizeof(Derived));

    Derived d{};
    size_t vptr_off = 0;
    size_t a_off = (size_t)((char*)&d.a - (char*)&d);
    size_t b_off = (size_t)((char*)&d.b - (char*)&d);
    size_t c_off = (size_t)((char*)&d.c - (char*)&d);
    printf("Derived layout vptr=%zu a=%zu b=%zu c=%zu sizeof=%zu\n",
           vptr_off, a_off, b_off, c_off, sizeof(Derived));
    return 0;
}
"""

with tempfile.TemporaryDirectory() as d:
    cpp = os.path.join(d, "vtbl.cpp")
    exe = os.path.join(d, "vtbl")
    with open(cpp, "w") as f:
        f.write(SRC)
    subprocess.run(["clang++", "-O2", "-std=c++20", "-Wall", cpp, "-o", exe], check=True)
    out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    print(out, end="")
PY
```

Read the first table down: `OneInt` to `OneVirtual` jumps from 4 to 16 bytes for one added
vptr, because the vptr's 8-byte alignment pads the whole struct up to a multiple of 8. The next
row is the point of the exercise: `ManyVirtual` declares four virtual methods instead of one and
is still 16 bytes, because the vtable grows, not the object. `VDerived`, inheriting its one
virtual method through `virtual` inheritance, also lands on 16 — a virtual base that is also the
primary base reuses the derived class's own vptr slot rather than adding a second one, so this
shape of virtual inheritance is not the size explosion the keyword suggests. The second table
shows the mechanism inside a plain `Derived : Base`: the vptr sits at offset 0, `Base`'s members
follow it, and `Derived::c` lands at offset 20 — inside `Base`'s own trailing padding — rather
than at 24, because a derived class's members start at the base's data size, not its padded
`sizeof`.

## Practise it

```bash
mlsys grade cpp-sizeof-with-vs-without-a-virtual-vptr
```

[That task](../tasks/cpp-sizeof-with-vs-without-a-virtual-vptr/task.md) gates a `virtual_sizeof`
function on `exact_match == 1.0` against five real polymorphic/plain struct pairs the driver
compiles itself, so the reference is genuine compiler output, not a hand-typed answer. The
shipped starter returns `0` unconditionally and fails the first pair immediately; the harder
mistake is forgetting the new alignment is `max(plain_align, 8)` rather than always `8`, which
passes pairs whose original alignment was already 8 and fails silently on the rest.

More of this area, in increasing difficulty:
[predict `sizeof` of a small struct](../tasks/cpp-predict-sizeof-of-a-small-struct/task.md) (no
virtual functions yet),
[single-inheritance object layout offsets](../tasks/cpp-single-inheritance-object-layout-offsets/task.md)
(the second table above, computed by you),
[multiple-inheritance this-pointer adjustment](../tasks/cpp-multiple-inheritance-this-pointer-adjustment/task.md)
(once there is more than one base),
[which override is called, 12 calls](../tasks/cpp-which-override-is-called-12-calls/task.md)
(dispatch correctness once slicing is involved),
[virtual-call indirection cost, modeled](../tasks/cpp-virtual-call-indirection-cost-modeled/task.md)
(the two-load cost above, counted across a call trace),
and [build a manual vtable dispatcher](../tasks/cpp-build-a-manual-vtable-dispatcher/task.md),
replacing `obj->compute(x)` with the raw pointer-chasing steps this page describes.

## Common mistakes

- **Assuming more virtual methods means more per-object storage.** `ManyVirtual` declares four
  virtual methods to `OneVirtual`'s one and both are 16 bytes — the vtable grows, the object's
  single pointer to it does not.
- **Expecting virtual inheritance to always add size.** `VDerived`, with one virtual base that
  is also its primary base, comes out the same 16 bytes as ordinary single inheritance; the
  well-known extra cost shows up with multiple virtual bases or diamonds, not this shape.
- **Casting between bases with `reinterpret_cast`.** Under multiple inheritance, converting a
  `Derived*` to a non-primary `Base*` needs a fixed byte offset added, because that base's
  subobject does not sit at offset 0. `static_cast` emits the adjustment; `reinterpret_cast`
  does not, and hands back a pointer to the wrong bytes.
- **Reading the layout as portable.** Every number above is a property of the Itanium C++ ABI
  that clang and gcc use on macOS and Linux; MSVC places vptrs and pads virtual bases
  differently, so a layout assumption ported across compilers can silently corrupt memory.

## Where else to practise this

Honest comparison, from the [full survey of what exists](../LANDSCAPE.md), which lists this
whole track as **153 tasks, some overlap** — one or two resources graded, covering part of the
area:

- **[CppQuiz.org](https://cppquiz.org/)** — 190 real-snippet questions scored on predicting
  exact output or UB, several on virtual dispatch and slicing, but read-and-predict only; you
  never write or compile a vtable yourself.
- **[HackerRank — C++ domain](https://www.hackerrank.com/domains/cpp)** — a small Inheritance
  subdomain covers virtual functions at introductory depth, with no object-layout coverage.
- **[LearnCpp.com](https://www.learncpp.com/)** — the default place most people first read *why*
  dynamic dispatch works this way, with a reveal-the-answer quiz and no compiler in the loop.
- **[Exercism — C++ track](https://exercism.org/tracks/cpp)** — real automated test-suite
  grading, but general-purpose C++ that touches classes incidentally, not vtable layout.
- **[Guru of the Week](https://herbsutter.com/gotw/)** — Herb Sutter's puzzle archive has issues
  built entirely around virtual-function surprises, each with a worked solution, no auto-check.

## References

1. Itanium C++ ABI, *Itanium C++ ABI: Runtime Object Model* — the specification defining vptr
   placement, vtable layout, and virtual-base offset rules that clang and gcc implement.
   https://itanium-cxx-abi.github.io/cxx-abi/abi.html
2. cppreference.com, *Virtual function specifier*. https://en.cppreference.com/w/cpp/language/virtual
