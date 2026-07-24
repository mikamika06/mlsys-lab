## Context

When using `std::vector` in C++, elements are stored in one contiguous heap
allocation. If you `push_back` and `size == capacity`, the vector must
**reallocate**: it allocates a new, larger block of memory and transfers
existing elements into it. If the element type has a `noexcept` move
constructor, the standard requires the vector to **move** the elements
into the new block instead of copying them (the strong exception-safety
guarantee only holds for `push_back`/`reserve` when moves cannot throw).

```cpp
struct Element {
    short   type;
    double* data;
    long    sizes[3];
};
```

`Element` here is move-only and its move constructor is instrumented to
count itself for real — you are not asked to hand-count moves, you are
asked to write the reallocation logic so that the *real* move constructor
gets called the right number of times.

## Task

Implement, in `solve.cpp`,

```cpp
long simulate_vector_pushes(int N, int initial_capacity, int growth_factor);
```

Simulate `N` `push_back` operations of freshly constructed temporary
`Element`s into a raw, manually managed growable array, following
`std::vector`'s real reallocation policy:

- Start with `size = 0`, `capacity = initial_capacity` (no allocation at
  all if `initial_capacity == 0`).
- Before each push, if `size == capacity`, reallocate: allocate a new raw
  buffer via `::operator new` (not `new Element[...]` — `Element` has no
  meaningful default state to double-construct) of
  `new_capacity = (capacity == 0 ? 1 : capacity * growth_factor)`
  elements, then move-construct (placement-`new` with `Element&&`) every
  existing element from the old buffer into the new one, in order,
  destroy the old elements, and release the old raw buffer via
  `::operator delete`.
- After capacity is sufficient, construct a temporary `Element` and
  move-construct it into slot `size` of the buffer, then `size += 1`.
- Before returning, destroy every live element and release the raw buffer
  — no leaks.

Return the number of reallocations performed. The driver reads the total
move count separately, from `g_move_count`, which `Element`'s own move
constructor increments — so every relocation in your code must go through
an actual `Element(Element&&)` call, not a manual field copy.

## Example

For `N=3, initial_capacity=0, growth_factor=2`:

- push 1: `size(0) == capacity(0)` → reallocate to capacity 1 (0 moved
  elements), then 1 insertion move.
- push 2: `size(1) == capacity(1)` → reallocate to capacity 2 (1 moved
  element), then 1 insertion move.
- push 3: `size(2) == capacity(2)` → reallocate to capacity 4 (2 moved
  elements), then 1 insertion move.

Total: 3 reallocations, `(0+1)+(1+1)+(2+1) = 6` moves.

## What the gate checks

The fixed driver (`main.cpp`) prints `sizeof(Element)` once, then runs
four fixed `(N, initial_capacity, growth_factor)` cases, printing each
case's reallocation count and the real observed move count. The gate is
an exact string match (`exact_match == 1.0`) against the reference's
printed output: wrong growth-policy arithmetic, copying instead of
moving, or double-counting/under-counting moves all change the numbers
and fail the gate.
