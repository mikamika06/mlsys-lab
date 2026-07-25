## Context

A general-purpose allocator (`malloc`) has to track every live
allocation independently, so each block it hands out is preceded by a
small bookkeeping **header** (size, free-list links, ...), and the
resulting block size is rounded up to the allocator's **alignment**
requirement. For `HEADER_BYTES` of header and `ALIGN_BYTES` of
alignment, one object of `obj_bytes` costs

$$
\mathrm{round\_up}(\mathrm{HEADER\_BYTES} + \mathrm{obj\_bytes},\ \mathrm{ALIGN\_BYTES})
$$

bytes of real memory. Allocate `count` objects one call at a time and
that overhead is paid **once per object**.

A **pool allocator** instead makes a single large allocation up front
and hands out slices of it. It still needs bookkeeping for that one
allocation, but only once, no matter how many objects live inside it:

$$
\mathrm{round\_up}(\mathrm{HEADER\_BYTES} + \mathrm{count} \cdot \mathrm{obj\_bytes},\ \mathrm{ALIGN\_BYTES})
$$

The gap between the two grows sharply as objects get smaller relative to
the header: a `4`-byte object with a `16`-byte header pays 5x its own
size in overhead under per-object `malloc`, but under a pool that
`16`-byte header gets amortized across every object in it.

## Task

Implement:

```cpp
long malloc_per_object_footprint(int count, int obj_bytes);
long pool_footprint(int count, int obj_bytes);
double footprint_ratio(int count, int obj_bytes);
```

Using the pinned `HEADER_BYTES` and `ALIGN_BYTES` (declared in
`sol.hpp`, defined in `main.cpp`):

- `malloc_per_object_footprint`: round `HEADER_BYTES + obj_bytes` up to
  the next multiple of `ALIGN_BYTES`, then multiply by `count`.
- `pool_footprint`: compute `HEADER_BYTES + count * obj_bytes` (one
  header for the whole pool), then round that total up to the next
  multiple of `ALIGN_BYTES`.
- `footprint_ratio`: `malloc_per_object_footprint(count, obj_bytes) /
  (double) pool_footprint(count, obj_bytes)`.

## Example

With `HEADER_BYTES = 16`, `ALIGN_BYTES = 16`, `count = 1000`,
`obj_bytes = 8`: per-object cost is `round_up(16 + 8, 16) = 32` bytes,
so `malloc_per_object_footprint = 1000 * 32 = 32000`. The pool's raw
size is `16 + 1000*8 = 8016`, already a multiple of `16`, so
`pool_footprint = 8016`. `footprint_ratio = 32000 / 8016 ≈ 3.992016` --
per-object `malloc` uses almost 4x the memory for the same 1000 objects.

## What the gate checks

`main.cpp` runs six fixed `(count, obj_bytes)` cases -- from many tiny
objects (where the header dominates) down to a single 1000-byte object
(where a pool buys nothing, `ratio = 1.0`) -- and prints all three
values for each. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. Forgetting the
per-object header, forgetting the alignment rounding, or (the most
tempting shortcut) computing both footprints with the same formula so
every ratio comes out `1.0` all change the printed numbers and fail.
