## Context

A move constructor exists to *transfer* ownership, not to duplicate it. The
classic bug is writing only half of the transfer:

```cpp
Buffer::Buffer(Buffer&& other) noexcept : id_(other.id_) {
    // forgot: other.id_ = 0;
}
```

Now two objects hold the same resource. Both destructors run, both release it,
and you get a double free — a crash that usually surfaces far away from the line
that caused it. `std::vector` exposes it immediately, because growing a vector
moves every element.

To make the failure visible instead of fatal, `sol.hpp` declares a tiny
instrumented heap: `heap::alloc` hands out unique ids, `heap::release` records a
double release instead of crashing, and `heap::live()` reports what is still
allocated. Releasing id `0` is a no-op, exactly like `delete nullptr`.

## Task

Edit `solve.cpp` so `Buffer` is a correct RAII type:

- **move constructor** — steal `other.id_` and leave the source owning nothing;
- **move assignment** — release what this object already owned, then steal;
- **copy constructor / copy assignment** — deep copy, i.e. a *new* allocation
  holding the same value;
- **self-assignment** (`a = a`) must leave `a` intact.

The destructor and the value accessor are already written.

## Example

```cpp
Buffer a(42);
Buffer b(std::move(a));
// b.value() == 42, a.id() == 0, heap::live() == 1
// leaving the scope must give heap::live() == 0 and heap::doubleFrees() == 0
```

## What the gate checks

`main.cpp` runs five scenarios — move construction, deep copy, move assignment,
copy assignment with a self-assignment, and a `std::vector` that reallocates —
and after each one prints the live allocation count and the double-free count.
The output is compared numerically against `ref.cpp`, compiled by the same
`clang++ -O2 -std=c++20`. The shipped buggy version scores `max_abs_err = 828`;
a correct one scores `0`.
