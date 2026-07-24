## Context

A GPU driver hands out opaque resource handles. Every handle that is acquired
must be released exactly once — no leaks (release count too low) and no
double-frees (release count too high). Manual `acquire`/`release` pairs are
fragile across early returns and exceptions, so the idiomatic C++ answer is a
move-only RAII wrapper with a custom deleter, exactly like `std::unique_ptr`
with a custom `Deleter`.

You are given a fake driver (fixed, in `sol.hpp`):

- `gpu_acquire()` returns a fresh, non-zero resource id.
- `gpu_release(id)` is the custom deleter; it increments the global counter
  `g_release_count` once per non-empty id (releasing the empty id `0` is a no-op).

The invariant to uphold is: after a scope that acquired $k$ resources exits by
**any** path (normal, move, or exception), `g_release_count == k`.

## Task

Implement the members of the move-only handle `GpuHandle` (declared in
`sol.hpp`) in `solve.cpp`:

- **Constructor** takes ownership of an id.
- **Destructor** releases the owned id exactly once via `gpu_release`.
- **Move constructor** steals ownership; the moved-from handle must own nothing.
- **Move assignment** releases the currently-owned id first, then steals the
  source's id and leaves the source empty.
- **`reset()`** releases the owned id immediately and leaves the handle empty,
  so the later destructor does not free it a second time.
- **`get()`** returns the owned id (`0` when empty).

Copy operations are already `= delete`d — the handle is move-only.

The fixed driver `main.cpp` runs five ownership scenarios, printing the release
count after each and their total.

## Example

For a correct implementation the driver prints:

```
1 1 2 1 1
total=6
```

- normal scope exit -> 1
- move construction (moved-from must not release) -> 1
- move assignment over a live handle (old id freed now, stolen id freed at exit) -> 2
- exception unwinding -> 1
- `reset()` then destructor (no double-free) -> 1

The starter releases nothing, so it prints `0 0 0 0 0` / `total=0`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with
`clang++ -O2 -std=c++20`, runs the binary, and requires an **exact match** of
the printed numbers against the reference. Any leak, double-free, or
release-on-move produces a different count and fails the gate.
