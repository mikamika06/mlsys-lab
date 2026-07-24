## Context

Choosing between `std::shared_ptr` and `std::unique_ptr` has real memory and
performance cost:

1. **`std::shared_ptr<T>`**:
   - Shared, reference-counted ownership.
   - Holds 2 raw pointers (managed object + control block) — real
     `sizeof(std::shared_ptr<T>)` is 16 bytes on this ABI.
   - The control block holds the strong and weak reference counters —
     modeled here as 16 bytes (two atomic counters).
   - Copying it, or destroying a copy, touches the reference count
     **atomically** (it has to be safe across threads): $2 \times
     \text{transfers}$ atomic operations for `transfers` copy/destroy
     round trips (one increment per copy, one decrement per destruction).

2. **`std::unique_ptr<T>`**:
   - Sole, unshared ownership.
   - Holds a single raw pointer — real `sizeof(std::unique_ptr<T>)` is 8
     bytes.
   - No control block, no atomics: moving it is just a pointer swap.

$$\text{atomic\_ops} = \begin{cases} 0 & \text{unique\_ptr} \\ 2 \times \text{transfers} & \text{shared\_ptr} \end{cases}$$

## Task

Implement

```cpp
OwnershipPlan optimize_ownership(bool is_sole_owned, int transfers, const TypeFacts& facts);
```

`TypeFacts` (declared in `sol.hpp`) carries the real, measured
`object_bytes`/`unique_ptr_bytes`/`shared_ptr_bytes` for the type in
question — `main.cpp` gets these from actual `sizeof(T)`,
`sizeof(std::unique_ptr<T>)`, and `sizeof(std::shared_ptr<T>)`, never a
hand-rolled ABI table. Return the `OwnershipPlan`:

- `pointer_type`: `"unique_ptr"` if `is_sole_owned`, else `"shared_ptr"`.
- `atomic_ops`: `0` for `unique_ptr`, `2 * transfers` for `shared_ptr`.
- `pointer_bytes`: `facts.unique_ptr_bytes` or `facts.shared_ptr_bytes`.
- `control_block_bytes`: `0` for `unique_ptr`, `16` for `shared_ptr`.
- `object_bytes`: `facts.object_bytes`.

## Example

For a sole-owned `struct { int; double; }` (`object_bytes = 16`):

```cpp
optimize_ownership(true, 5, facts);
// -> {"unique_ptr", 0, 8, 0, 16}
```

## What the gate checks

Before calling your function at all, `main.cpp` **proves** the
"2 atomic ops per transfer" model against a real `std::shared_ptr<T>`: it
runs `transfers` real copy/destroy cycles and checks `use_count()` lands
back at its starting value — confirming each cycle really is one atomic
increment (the copy) paired with one atomic decrement (the copy going out
of scope), not an assumption. It then measures real `TypeFacts` for 3
different struct types and asks your `optimize_ownership` for 4 scenarios
(sole-owned and shared, with different transfer counts and types),
printing each resulting plan. The grader compiles your `.cpp` with the
real local `clang++` and requires all 4 printed lines to match the
reference's exactly ($\mathrm{exact\_match} = 1.0$).
