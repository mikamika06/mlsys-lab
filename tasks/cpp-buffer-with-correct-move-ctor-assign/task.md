## Context

Classes that own a heap resource must correctly implement the "Rule of Five": a
destructor, copy constructor, move constructor, copy assignment operator, and
move assignment operator. Get this wrong and you get double frees, leaks, or
needless deep copies. Get it right and:

- copying an object allocates a fresh resource and deep-copies the contents;
- moving an object is *cheap* — it steals the source's resource and leaves the
  source empty, without ever touching the data;
- self-assignment (`x = x`, possibly through an alias) must be a safe no-op,
  not a free-then-use-after-free.

## Task

Implement the five special members of `Buffer` (declared in `sol.hpp`) using the
harness's instrumented fake heap: `tracked_alloc(size)` returns a fresh nonzero
id, `tracked_free(id)` retires it, `tracked_deep_copy(dst_id, src_id, size)`
records a deep copy. `Buffer` has two public fields: `ptr` (the id owned, or `0`
for "owns nothing") and `size`.

- `Buffer(long size_)` — construct: `ptr = tracked_alloc(size_)`, `size = size_`.
- `~Buffer()` — destroy: `tracked_free(ptr)` if `ptr != 0`.
- `Buffer(const Buffer& other)` — copy ctor: allocate `other.size` bytes, then
  `tracked_deep_copy` from `other.ptr` into the new id.
- `Buffer(Buffer&& other) noexcept` — move ctor: steal `other.ptr`/`other.size`
  directly (no `tracked_alloc`, no `tracked_deep_copy`), then null `other` out
  (`ptr = 0`, `size = 0`).
- `operator=(const Buffer& other)` — copy assign: if `&other == this`, do
  nothing. Otherwise free the current `ptr` (if nonzero), allocate
  `other.size` bytes, and deep-copy from `other.ptr`.
- `operator=(Buffer&& other) noexcept` — move assign: if `&other == this`, do
  nothing. Otherwise free the current `ptr` (if nonzero), steal
  `other.ptr`/`other.size`, then null `other` out.

## Example

```cpp
Buffer b1(100);          // alloc
Buffer b2(b1);            // copy ctor: alloc + deep_copy; b1 untouched
Buffer b3(std::move(b1)); // move ctor: steal; b1.ptr == 0, b1.size == 0

b2 = b3;                  // copy assign: free b2's old buffer, alloc, deep_copy
b3 = std::move(b2);       // move assign: free b3's old buffer, steal from b2
```

## What the gate checks

The driver runs a fixed sequence of constructions, a clone, a move
construction, a copy assignment, a move assignment, a self copy-assignment,
and a self move-assignment, printing the running `allocs` / `frees` /
`deep_copies` totals plus each buffer's null-ness and size after every step.
The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20` and requires

$$ \mathrm{exact\_match} = 1.0 $$

against the reference. In particular: moving must never call
`tracked_deep_copy`, a moved-from buffer must read `ptr == 0` and `size == 0`
afterward, both assignment operators must free the receiver's *old* resource
before taking the new one, and self-assignment (through an aliasing pointer,
so the compiler can't just elide it) must leave allocs/frees/deep_copies
completely unchanged. Missing any one of these shifts a later count and the
whole printed trace stops matching bit-for-bit.
