## Context

CPython's small-object allocator groups memory into arenas, pools, and blocks. An arena contains a fixed number of pools, and pools contain blocks that can be allocated or returned to a free list.

This task uses a simplified deterministic model of that hierarchy. An arena is resident while it contains at least one pool that has been created. Pools are created on demand when allocation needs more capacity. A free operation returns a block to its pool, and a completely empty pool can become reusable, but arenas remain resident.

The model tracks three values:

- $P$: the number of pools per arena.
- $B$: the number of blocks in each pool.
- $U$: the number of used blocks.

A pool is usable if its number of used blocks is less than $B$. A new arena is needed only when all existing pools are full.

## Task

Implement `arena_occupancy(trace, pools_per_arena, blocks_per_pool)`:

```python
def arena_occupancy(trace, pools_per_arena, blocks_per_pool):
    ...
```

The input `trace` is a list of operations. Each operation is a tuple:

- `("alloc", id)` allocates one block and associates it with integer `id`.
- `("free", id)` frees the block previously allocated with that `id`.

Return a list containing the number of resident arenas after every operation.

Allocation chooses the first usable pool with free capacity. If no usable pool exists, create a new pool in the first arena with available pool slots. If all arenas are full, create a new arena containing the new pool.

A pool's identity is not returned. Only the number of resident arenas is required.

## Example

```python
trace = [
    ("alloc", 1),
    ("alloc", 2),
    ("free", 1),
    ("alloc", 3),
]

arena_occupancy(trace, 2, 2)
# [1, 1, 1, 1]
```

With two pools per arena and two blocks per pool, all operations fit inside the first arena.

## What the gate checks

The gate runs several allocation and free traces and compares the returned arena counts against an internal reference model of the allocator rules.

The `exact_match` score must be $1.0$. The implementation must correctly maintain pool usage, block ownership, and arena growth decisions.
