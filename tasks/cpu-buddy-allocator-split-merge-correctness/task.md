## Context

A **buddy allocator** manages a fixed-size arena in power-of-two blocks. `alloc(size)` rounds `size` up to the smallest power-of-two block that fits, and if no block of exactly that size is free, it finds the smallest available *larger* free block and **splits** it in half repeatedly -- each split creates two same-size "buddies" -- until a block of the right size exists.

`free(offset)` is only half the job if it just marks the block free again: without **merging**, the arena permanently fragments into ever-smaller pieces. The other half is checking whether the freed block's buddy is *also* a whole free block; if so, merge them back into their parent block, and repeat the check one level up, as far as it will go.

## Task

`BuddyAllocator` (declared in `sol.hpp`) manages a 256-byte arena with a 16-byte minimum block size (5 levels: 256, 128, 64, 32, 16). Fix `BuddyAllocator::free` in `solve.cpp` so that after marking a block free, it walks back up the levels merging with the buddy at each level for as long as that buddy is also a whole free block (`alloc` is already correct as shipped).

## Example

```cpp
BuddyAllocator bud;
int a = bud.alloc(20);   // rounds up to 32 bytes, splits 256->128->64->32
int b = bud.alloc(200);  // rounds up to 256 bytes -- fails (-1), arena is fragmented
bud.free(a);
int c = bud.alloc(256);  // now succeeds -- but ONLY if free() actually merged
                           // a's block back up through 32->64->128->256
```

## What the gate checks

`main.cpp` runs a fixed sequence of `alloc`/`free` calls -- exercising an initial split cascade, reuse of freed space, a partial free that should merge into a bigger block, an oversized allocation that should correctly fail while other blocks are still live, and finally freeing everything and asking for the whole 256-byte arena back -- and prints every returned offset. Your printed output is compared against `ref.cpp`, compiled and run the same way: `max_abs_err <= 1e-9`. Skipping the buddy-merge step lets small allocations still succeed by chance, but the very last allocation (the whole arena, only obtainable if every earlier free chained all the way back up) fails where the reference succeeds.
