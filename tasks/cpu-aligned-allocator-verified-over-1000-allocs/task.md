## Context

A **bump allocator** hands out memory from a fixed arena by keeping a single running offset: each allocation rounds the current position up to the requested alignment, reserves `size` bytes there, and advances the offset past it. It never frees individual allocations, but it's fast and exactly reproducible — no free-list bookkeeping, no fragmentation surprises.

Aligning correctly means rounding the actual runtime **pointer address** up to the next multiple of `align` (a power of two) — not just the offset from the arena's start, since the arena's own base address is not guaranteed to already be aligned to every `align` you might be asked for.

## Task

Implement:

```cpp
void bump_init(BumpState& s, unsigned char* arena, std::size_t arena_size);
void* aligned_alloc_bump(BumpState& s, std::size_t size, std::size_t align);
```

`aligned_alloc_bump` must round `reinterpret_cast<uintptr_t>(s.arena) + s.offset` up to the next multiple of `align`, reserve `size` bytes there, advance `s.offset` to just past the end of that reservation (measured from the start of the arena), and return the aligned pointer. If the reservation would not fit within `s.arena_size` bytes, return `nullptr` and leave `s.offset` unchanged.

## Example

With an empty arena (`offset = 0`) and `align = 16`: if the arena's own base address is not already a multiple of `16`, the very first allocation still starts at some non-zero offset — whatever it takes to reach the next 16-byte-aligned address — not at offset `0`.

## What the gate checks

`main.cpp` performs `1000` allocations against a real `1 MiB` byte arena, cycling through alignments `1, 2, 4, 8, 16, 32, 64` and sizes `1..37` bytes in a fixed deterministic pattern, and prints the count of non-null results, the count of correctly-aligned pointers, the count of non-overlapping allocations, and the final `offset`. The candidate's full stdout is compared byte-for-byte (`exact_match = 1.0`) against the reference's. Rounding the *offset* instead of the *actual address* looks correct on many toolchains (whose static arrays happen to land on aligned addresses) but silently breaks the moment the arena isn't already aligned to the requested boundary.
