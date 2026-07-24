## Context

When a C++ function hands a heap buffer back to a caller — the pattern behind
a pybind11 function returning a numpy array whose data pointer is
C++-allocated and whose lifetime is tied to a capsule "base" object — nobody
calls `delete`/`free` explicitly. The buffer is freed *automatically*,
exactly once, whenever the object that owns it is destroyed: the last
`Capsule` (or, in the real Python binding, the last reference to the
capsule) to hold the pointer is responsible for it.

That guarantee only holds if ownership transfer is implemented correctly:

- **Move-construct** (the buffer survives being returned by value, or a
  `std::vector` reallocating and moving existing elements to new storage):
  the new object must adopt the pointer, and the old object must let go of
  it — otherwise both objects' destructors free the *same* pointer (a
  double-free).
- **Move-assign onto a live object**: the assignment target may already own
  a buffer. That buffer must be freed *before* the target adopts the new
  one, or it leaks (nobody ever frees it).
- **Copy is not allowed** at all — two capsules must never believe they each
  own the same buffer.

## Task

Implement, for the `Capsule` type declared in `sol.hpp`:

```cpp
Capsule(Capsule&& other) noexcept;               // adopt other's buffer, empty other
Capsule& operator=(Capsule&& other) noexcept;     // free own buffer first, then adopt + empty other
~Capsule();                                       // free the owned buffer, if any
Capsule make_capsule(int n, int mult);            // allocate n bytes, fill byte i with (unsigned char)(i*mult)
```

Every buffer must be allocated with `arena_alloc` and released with
`arena_free` — these are declared in `sol.hpp` and instrumented by the
driver, which counts allocations against frees. "Empty" means
`data == nullptr` (so a later destructor call on that object is a no-op,
since `arena_free(nullptr)` does not count as a free).

## Example

```cpp
Capsule a = make_capsule(4, 1);   // a.data = {0, 1, 2, 3}
Capsule b = std::move(a);         // b now owns {0,1,2,3}; a.data == nullptr
// a's destructor: no-op (nothing to free)
// b's destructor: frees {0,1,2,3} -- exactly once
```

## What the gate checks

The driver runs a fixed scenario: five capsules built inside a growing
`std::vector` (forcing move-construction on reallocation), a move-assign
onto an already-populated capsule, a temporary destroyed at the end of its
own scope, and a separate move-construct — then destroys everything and
prints each surviving capsule's size and byte contents, followed by the
final `alloc=<n> free=<n>` counts. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed capsule's contents AND alloc/free counts match the reference}
$$

A move that doesn't empty its source produces a real double-free against
the real allocator (crash, not just a wrong number); a move-assign that
doesn't free the target's existing buffer first leaves `alloc > free` — both
are visible in the printed output, not hidden behind a simulator.
