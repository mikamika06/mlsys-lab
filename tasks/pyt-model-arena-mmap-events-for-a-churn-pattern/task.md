## Context

CPython's small-object allocator groups small allocations into pools and arenas. This task
uses a simplified allocator model that measures object sizes with the running interpreter.

Each allocation request $n$ is converted into a size class using
`sys.getsizeof(bytearray(n))`. Allocations with the same measured size class share pools.
A pool can hold $B=32$ blocks. An arena can contain $P=64$ pools.

When no free pool exists, the model creates a new arena:

$$
\text{mmap events} = \text{number of times a new arena is created}.
$$

When an arena has no remaining active pools after blocks are returned, it is released:

$$
\text{munmap events} = \text{number of released arenas}.
$$

The allocator model processes allocations in order and then frees them in reverse order,
which represents a churn pattern.

## Task

Implement `arena_mmap_events(sizes)`:

```python
def arena_mmap_events(sizes: list[int]) -> tuple[int, int]:
    ...
```

The function receives allocation request sizes and returns:

```python
(mmap_events, munmap_events)
```

Implement the following deterministic model:

- Convert every request using `sys.getsizeof(bytearray(size))`.
- Maintain separate pools for each resulting size class.
- A pool starts with $32$ available blocks.
- When a size class needs another pool, obtain one from the arena pool supply.
- If no arena has a free pool, create an arena and increment `mmap_events`.
- After all allocations, free objects in reverse order.
- When a pool becomes empty, return it to the arena.
- When an arena has no active pools, release it and increment `munmap_events`.

## Example

```python
arena_mmap_events([1, 1, 1, 1000])
# returns (1, 1)
```

The exact result is based on the active CPython interpreter's object size layout.

## What the gate checks

The gate computes the expected result using an independent reference implementation that
also queries `sys.getsizeof(bytearray(n))`.

The candidate must return exactly the same `(mmap_events, munmap_events)` pair for all
tested churn patterns.
