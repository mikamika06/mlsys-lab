## Context

A blocked kernel's inner loop revisits the same tile many times (once
per step of the reduction dimension it is blocking over). If the tile's
whole footprint fits in L2, only the first pass pays real DRAM/L3
traffic — every later pass is served from L2. If the tile is too big,
every single pass thrashes the cache and pays that traffic all over
again, because nothing survives between passes.

## Task

Implement

```cpp
int pick_resident_tile(int tile_b0, int tile_b1, int passes, long* out_misses);
```

A tile of side length `B` is 3 contiguous `B x B` float arrays (`3*B*B*4`
bytes total). For each candidate `tile_b0` and `tile_b1`: `reset_cache()`,
then touch every address of that tile's 3 arrays (see `sol.hpp` for the
exact address formula) via `touch_byte()`, `passes` times in a row (same
addresses each pass). Write the total miss count over all `passes`
repetitions into `out_misses[0]` (for `tile_b0`) and `out_misses[1]`
(for `tile_b1`). Return `0` or `1`: the id of whichever tile produced
fewer total misses.

## Example

With a 4096-byte L2, `tile_b0 = 16` needs `3*16*16*4 = 3072` bytes —
fits, so only the first of 5 passes misses (`48` misses, `0` per later
pass). `tile_b1 = 32` needs `3*32*32*4 = 12288` bytes — 3x too big, so
every one of the 5 passes thrashes and re-misses in full. `tile_b0`
wins by a wide margin even though it does less useful work per pass.

## What the gate checks

`exact_match`: the driver prints the winning id and both miss counts for
one fixed pair of tile sizes and 5 passes. Getting either miss count
wrong (wrong address formula, resetting the cache between passes instead
of only between tiles, or not repeating for all `passes`) changes the
printed line even if the winning id happens to still be right; a starter
returning `0, 0` fails outright since the real counts are `48` and
`960`.
