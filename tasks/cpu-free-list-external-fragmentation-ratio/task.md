## Context

A free-list allocator doesn't just track *how much* memory is free — the
*shape* of that free memory matters just as much. Two allocators can both
be sitting on 400 free bytes: one as a single contiguous block that can
satisfy any request up to 400 bytes, the other as forty scattered
10-byte scraps that can't satisfy a request bigger than 10. The second
one is externally fragmented, and the standard way to quantify *how*
fragmented is the ratio of free bytes stranded outside the single largest
free block:

$$
\text{ratio} = \frac{\text{total\_free} - \text{largest\_free\_block}}{\text{total\_free}}
$$

`0` means every free byte is part of one contiguous block (as good as it
gets); close to `1` means the free memory is scattered into many small
pieces, none of them individually useful.

## Task

Implement

```cpp
double external_fragmentation_ratio(long heap_bytes,
                                     const int* op_types, const int* op_sizes, const int* op_ids,
                                     int num_ops);
```

Simulate a `heap_bytes`-byte heap, starting as one free block covering
`[0, heap_bytes)`. Run the `num_ops` operations in order:

- **`op_types[i] == 0` (ALLOC)**: request `op_sizes[i]` bytes. Using
  **first-fit** (scan blocks in address order), find the first FREE block
  big enough — guaranteed to exist in this task's scenarios. If it's an
  exact match, mark it USED. If it's bigger, **split** it: carve a used
  block of exactly the requested size off its low-address end, leaving
  the remainder as a new, smaller free block right after it.
- **`op_types[i] == 1` (FREE)**: `op_ids[i]` is the op-index of the ALLOC
  being released. Mark that block FREE, then **coalesce**: check both the
  physically-adjacent block before it and the one after it in address
  order, and merge with either (or both) if free.

After the last op, return the external fragmentation ratio over whatever
free blocks remain (`0.0` if nothing is free).

## Example

A 500-byte heap fully allocated as five blocks, then two of them
(non-adjacent to each other, separated by a still-used block) are freed:
the free bytes end up as two separate blocks, say sizes `150` and `240`.
Total free is `390`, the largest block is `240`, so the ratio is
`(390 - 240) / 390 = 150 / 390 \approx 0.3846`.

## What the gate checks

The driver runs two independent scenarios — a 1000-byte heap where three
non-adjacent blocks are freed (no coalescing occurs, three separate free
blocks remain), and a 500-byte heap where two adjacent frees coalesce
into one run while a third, separated free block stays apart, and then a
final small allocation splits the larger of the two — and prints both
ratios. The grader compiles `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires

$$
\max_i |\,\text{printed}_i - \text{reference}_i\,| \le 10^{-6}
$$

The reference prints `ratioA=0.454545 ratioB=0.314286`. A stub that
returns `0.0` for everything — or one that computes total free bytes
correctly but skips coalescing (so it never finds the single largest
merged block, only individual un-merged fragments) — produces the wrong
ratio for at least one scenario and fails the gate.
