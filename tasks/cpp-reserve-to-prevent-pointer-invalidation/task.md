## Context

`std::vector` stores elements in one contiguous block. When an insertion
would exceed the current capacity, the vector **reallocates**: allocate a
bigger block, move/copy every existing element into it, free the old block.
That invalidates every pointer, reference, and iterator into the old block
— including ones you cached before the insertion that triggered it.

Calling `v.reserve(N)` before any insertions pre-allocates room for at least
`N` elements in one shot, so none of the following `push_back`s (as long as
you don't exceed `N`) ever reallocates — every pointer you take into the
vector stays valid.

## Task

Implement `grow_vector(int n_elements, bool reserve_first)` (declared in
`sol.hpp`), returning a `GrowthResult { realloc_count, final_capacity,
pointers_valid }`:

- Build a real `std::vector<Item>` (`Item` is a fixed 2-field struct).
- If `reserve_first`, call `v.reserve(n_elements)` before the loop.
- `push_back` `n_elements` real items, one at a time.
- Track how many times `v.data()` changes address across the pushes AFTER
  the first one (the very first push's own allocation isn't a
  "re"-allocation — there was nothing before it to invalidate).
- `pointers_valid`: whether the address observed right after the first
  push is still `v.data()`'s address once the loop ends (trivially `true`
  for `n_elements == 0`, nothing was ever pushed).

The fixed driver (`main.cpp`) then prints `realloc_count`,
`v.capacity()`, `capacity * sizeof(Item)`, and `pointers_valid` — real
numbers from a real standard-library `std::vector`, not a modeled growth
formula.

## Example

For the fixed driver's six scenarios, the correct run prints
(`realloc_count final_capacity allocated_bytes pointers_valid`):

```
0 10 160 1
4 16 256 0
0 0 0 1
0 100 1600 1
7 128 2048 0
0 1024 16384 1
```

Every `reserve_first=true` case shows `realloc_count=0` and
`final_capacity` exactly equal to `n_elements`. Without `reserve`, this
machine's standard library grows by doubling from `1`: reaching 10 elements
takes the sequence `1, 2, 4, 8, 16` — 4 reallocations, landing on capacity
16 (not 10); reaching 100 takes 7 doublings to land on 128.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of all six printed lines against
the same driver linked with `ref.cpp`. Missing the `reserve()` call, or
miscounting reallocations (e.g. counting the very first allocation as one),
changes a printed line and fails the gate.
