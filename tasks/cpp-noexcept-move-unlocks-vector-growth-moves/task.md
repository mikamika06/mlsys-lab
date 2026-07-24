## Context

When `std::vector<T>` exceeds its capacity during a push, it allocates a bigger buffer (this platform's libc++ doubles capacity: $0 \to 1 \to 2 \to 4 \to 8 \to \dots$) and relocates every existing element from the old buffer to the new one.

To satisfy the strong exception guarantee, `std::vector` uses `std::move_if_noexcept` for that relocation:

- If `T`'s move constructor is `noexcept`, the vector relocates via **move construction** — zero copies.
- If it is *not* `noexcept`, the vector falls back to **copy construction** for every existing element, so that a throwing move can never leave the vector in a corrupted state.

## Task

Implement `GrowthCounts simulate_vector_growth(int element_size, int n_pushes, bool move_is_noexcept)`. Push `n_pushes` elements, one at a time via `emplace_back()`, onto a **real** `std::vector<Elem>` that you build, where `Elem` is exactly `element_size` bytes (`8` or `16` for the cases this is graded against) and whose move constructor is declared

```cpp
Elem(Elem&& other) noexcept(move_is_noexcept) { ... }
```

Wire `Elem`'s copy constructor, move constructor, and destructor to increment `g_counters.copies` / `g_counters.moves` / `g_counters.destructions` (declared in `sol.hpp`). Reset `g_counters` before pushing, then report the counts it actually observed, plus `total_alloc_bytes` (the sum of `capacity() * sizeof(Elem)` at every point `capacity()` changed) and `final_capacity` (`capacity()` after all pushes).

## Example

For `element_size = 16`, `n_pushes = 10`, `move_is_noexcept = true`: capacity grows $0\to1\to2\to4\to8\to16$, relocating $0+1+2+4+8=15$ old elements at those five reallocations — all as moves since the move constructor is noexcept, so `copies = 0`, `moves = 15`, `destructions = 15`, `final_capacity = 16`.

## What the gate checks

`main.cpp` runs seven fixed `(element_size, n_pushes, move_is_noexcept)` cases and prints every field of the returned `GrowthCounts` for each. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's, whose numbers come straight from a real `std::vector<Elem>`'s real reallocation behavior — nothing here is simulated. Writing the move constructor without a conditional `noexcept` specifier (a common real-world mistake: adding a move constructor but forgetting `noexcept`, so the compiler treats it as potentially throwing) silently forces the vector to copy on every reallocation regardless of what `move_is_noexcept` asked for.
