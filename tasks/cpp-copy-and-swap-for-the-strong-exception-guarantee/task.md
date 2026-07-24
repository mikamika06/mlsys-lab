## Context

The **copy-and-swap idiom** implements assignment with the **strong
exception guarantee**: if any exception is thrown during the operation, the
target object is left completely unchanged. The pattern:

```cpp
T& operator=(T rhs) {   // rhs is a COPY, made while passing the argument
    swap(*this, rhs);   // nothrow swap of the copy into *this
    return *this;       // old *this, now living in rhs, is destroyed here
}
```

The copy of the source happens while constructing the by-value parameter
`rhs` — *before* `operator=`'s body runs at all. If that copy throws (out of
memory, a failed deep-copy, ...), the exception propagates out of the
function call and `operator=`'s body never executes, so `*this` was never
touched. `swap` itself must be `noexcept` (a plain pointer/int exchange, no
allocation), so once the copy has succeeded, nothing can go wrong on the way
to `*this` taking on the new value. Either the assignment fully completes,
or `*this` is byte-identical to what it was before — no in-between state is
ever observable.

## Task

`ByteBuffer` (declared in `sol.hpp`) owns a heap-allocated array of bytes. A
`ByteBuffer` built with `is_poisoned = true` must **throw
`std::runtime_error`** when it is copy-constructed, instead of copying —
this is how the driver simulates a copy that fails partway through a real
program (an allocation failure, a corrupted source, ...). Implement:

```cpp
ByteBuffer(const unsigned char* bytes, int n, bool is_poisoned);  // owns a deep copy of bytes[0..n)
ByteBuffer(const ByteBuffer& other);                               // deep copy; THROWS if other.poisoned
~ByteBuffer();                                                     // frees the owned buffer
void swap_with(ByteBuffer& other) noexcept;                        // exchange contents, no allocation
ByteBuffer& operator=(ByteBuffer rhs);                              // copy-and-swap: swap rhs into *this
```

`operator=` takes its argument **by value** on purpose (see the Context
pattern above) — do not change that signature. Its body should do nothing
but `swap_with(rhs)` and return `*this`.

## Example

```cpp
ByteBuffer target(t0, 4, /*is_poisoned=*/false);
ByteBuffer ok_source(s0, 4, /*is_poisoned=*/false);
target = ok_source;      // succeeds: target now holds s0's bytes

ByteBuffer bad_source(s0, 4, /*is_poisoned=*/true);
target = bad_source;     // throws std::runtime_error; target is UNCHANGED
```

## What the gate checks

The driver runs both scenarios: a normal assignment, and an assignment where
the source is poisoned (wrapped in a `try`/`catch` that records whether it
threw). It prints the caught-exception flag and the target's bytes after
each scenario. The grader compiles `solve.cpp` with
`clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed flag and every printed byte matches the reference}
$$

An implementation that copies the source's bytes directly into `*this`
(instead of going through a by-value parameter and swap) might get the
successful-assignment case right, but on the throwing case it either doesn't
throw at all, or has already started overwriting `*this` before the
exception fires — leaving the target in a corrupted, partially-overwritten
state instead of byte-identical to before.
