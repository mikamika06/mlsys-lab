## Context

Under the hood, `std::vector` dynamically manages memory. When an element is
added via `push_back` and the current `size` equals the `capacity`, the
vector must reallocate: a new, larger buffer, copy/move the existing
elements, deallocate the old one. Different STL implementations use
different growth factors (GCC's `libstdc++` typically uses 2.0; Clang's
`libc++` and MSVC use 1.5).

```cpp
struct VectorHeader { int size; int capacity; void* data; };
```

## Task

Implement `grow_capacity(int capacity, double growth_factor)` (declared in
`sol.hpp`): given the vector's current capacity right before a reallocation,
return its new capacity —

$$\text{new\_capacity} = \max(1, \lfloor \text{capacity} \times \text{growth\_factor} \rfloor_{\text{trunc}})$$

where the cast to `int` truncates toward zero, exactly like C++'s own
`double -> int` conversion.

The fixed driver (`main.cpp`) starts a simulated vector at `size = capacity =
0` and runs `n_pushes` simulated `push_back()`s for three
`(n_pushes, growth_factor)` scenarios: whenever `size == capacity` it calls
your `grow_capacity`, then increments `size` and prints the capacity **after**
that push.

## Example

For `growth_factor = 2.0`: capacity starts at `0`; the first push has
`size == capacity` (`0 == 0`), so `grow_capacity(0, 2.0) = max(1, 0) = 1`.

For the fixed driver's three scenarios, the correct run prints:

```
1 2 4 4 8 8 8 8 16 16
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
sizeof(VectorHeader)=16
```

Growth factor 1.5 (and 1.25) are a real gotcha: `grow_capacity(1, 1.5)` is
`(int)(1.5) = 1` — capacity gets stuck at 1 forever once it reaches 1,
because truncation with a sub-2.0 factor can fail to make any progress at
all from a small capacity. That's not a bug in the driver; it's exactly why
`libstdc++`-style implementations avoid factors below 2, or floor-guard the
growth so it always at least increases by 1.

The starter (missing the `max(1, ...)` floor and returning `0` unconditionally)
never grows the vector past capacity `0`, printing all-zero traces instead.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of all three printed capacity
traces (plus the fixed `sizeof(VectorHeader)` line) against the same driver
linked with `ref.cpp`. Getting the truncation or the `max(1, ...)` floor
wrong desyncs the trace from the very first push and fails the gate.
