## Context

A bump (arena) allocator is about as simple as allocators get: hand out
memory from one fixed block by advancing a single cursor. There's no
free-list, no fragmentation bookkeeping — allocation is just "round the
cursor up to the requested alignment, and move it forward by the requested
size." The two ways to get it wrong are: forgetting the alignment rounding
(misaligned allocations), and letting a failed allocation (one that would
run off the end of the arena) still consume space.

## Task

Implement, in `solve.cpp`:

- `bump_reset()` — reset the internal cursor to `0`.
- `bump_alloc(long size, long align, long arena_bytes)`:
  1. Round the current cursor up to the next multiple of `align` (a power
     of two).
  2. If `rounded_cursor + size > arena_bytes`, the allocation does not fit:
     return `-1` and leave the cursor **unchanged**.
  3. Otherwise, advance the cursor to `rounded_cursor + size` and return
     `rounded_cursor` as this allocation's offset.

## Example

For a 128-byte arena and the request sequence
`(size, align) = (10,1), (7,4), (20,8), (3,1), (50,16), (100,1), (20,1)`:

```
cursor=0   -> align 1  -> offset 0,  cursor becomes 10
cursor=10  -> align 4  -> round up to 12 -> offset 12, cursor becomes 19
cursor=19  -> align 8  -> round up to 24 -> offset 24, cursor becomes 44
cursor=44  -> align 1  -> offset 44, cursor becomes 47
cursor=47  -> align 16 -> round up to 48 -> offset 48, cursor becomes 98
cursor=98  -> size 100 -> 98+100=198 > 128 -> DOESN'T FIT -> -1, cursor stays 98
cursor=98  -> align 1  -> offset 98, cursor becomes 118
```

The fixed driver (`main.cpp`) runs exactly this sequence, prints every
returned offset, then independently checks every pair of successful
allocations for interval overlap. The correct run prints:

```
0 12 24 44 48 -1 98
overlaps=0
```

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the seven printed offsets and
the overlap count against the same driver linked with `ref.cpp`. Skipping
the alignment rounding, or letting a failed (`-1`) allocation advance the
cursor anyway, changes a later offset (or introduces a real overlap) and
fails the gate.
