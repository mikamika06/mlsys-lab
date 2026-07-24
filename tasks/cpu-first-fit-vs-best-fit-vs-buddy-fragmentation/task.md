## Context

Three classic allocation policies handle the same alloc/free trace very
differently:

- **First-fit** scans the free list in address order and takes the first
  block that's big enough.
- **Best-fit** scans every free block and takes the smallest one that's
  still big enough, to waste as little as possible on this one request.
- **Buddy** only ever splits a block into two equal halves and can only
  merge a freed block back with its one designated buddy — never with an
  arbitrary free neighbor, even if they happen to be physically adjacent.

All three can end up with the *same total* number of free bytes after a
trace (that only depends on how many bytes were allocated vs. freed) but
very different **external fragmentation**:

$$\text{fragmentation} = \text{total\_free\_bytes} - \text{largest\_contiguous\_free\_block}$$

This is the number of free bytes that exist but can't actually satisfy a
request for the largest available size, because they're stuck in smaller,
non-contiguous (or, for buddy, non-mergeable) pieces.

## Task

Implement `fragmentation_after_trace(op_kind, op_arg, num_ops, out)`. Build
three independent allocators, each managing its own fresh 256-byte arena
(see `sol.hpp` for the exact split/coalesce rules of each), replay the
*same* fixed trace through all three, and write each one's final
fragmentation into `out[0]` (first-fit), `out[1]` (best-fit), `out[2]`
(buddy).

## Example

Trace: alloc four 40-byte blocks (filling 160 of 256 bytes), free the 1st
and 3rd (two separate 40-byte holes), alloc 70 bytes (only the untouched
96-byte remainder is big enough), alloc 20 bytes, free the 2nd original
block.

First-fit and best-fit end up with the same 126 total free bytes (freeing
and re-allocating the same sizes), but best-fit's smallest-block-first
choice for the 20-byte request leaves a tiny 6-byte sliver stranded far
from the rest, while merging the freed 2nd block into its neighbors
produces one big 100-byte run for first-fit vs. one bigger 120-byte run
for best-fit — so best-fit ends up *less* fragmented here (`6` vs `26`).
Buddy fares far worse (`96`): it has 160 free bytes total (two 64-byte
blocks are free), but neither is the other's buddy, so buddy can't merge
them into anything bigger than 64 bytes even though 128 contiguous-in-value
free bytes exist across the two.

## What the gate checks

`exact_match`: the driver prints all three fragmentation numbers from one
fixed 9-op trace. Getting first-fit's or best-fit's block-selection rule
backwards, skipping neighbor coalescing on `free()`, or letting buddy merge
with a non-buddy free neighbor, all change at least one of the three
printed numbers.
