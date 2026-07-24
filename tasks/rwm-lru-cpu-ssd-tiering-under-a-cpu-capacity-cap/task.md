## Context

A tiered storage system keeps frequently reused chunks in a fast CPU tier while spilling less useful chunks to a slower SSD tier. The CPU tier has a fixed byte capacity, so admitting a new chunk may require evicting existing chunks.

A common policy is least recently used (LRU). The policy maintains an ordering of CPU-resident chunks by recency. When a chunk is accessed, it becomes the most recently used item. If a new chunk is loaded and the capacity would be exceeded, the least recently used chunks are evicted until the byte constraint is satisfied.

For a set of CPU-resident chunks $R$, the total reuse benefit can be modeled as

$$S(R) = \sum_{i \in R} s_i,$$

where $s_i$ is the saved GPU token cost from keeping chunk $i$ in CPU memory instead of fetching it again from SSD.

## Task

Implement `tier_lru(trace, sizes, savings, max_cpu_bytes)`:

```python
def tier_lru(
    trace: list[int],
    sizes: dict[int, int],
    savings: dict[int, int],
    max_cpu_bytes: int,
) -> tuple[list[int], int]:
    ...
```

The input `trace` is an ordered sequence of chunk accesses. `sizes[id]` gives each chunk's CPU memory size in bytes and `savings[id]` gives its GPU-token savings when resident.

Start with an empty CPU tier. Process accesses in order using an LRU policy:

- If an accessed chunk is already in CPU, mark it as most recently used.
- Otherwise, load it into CPU.
- If CPU usage exceeds `max_cpu_bytes`, evict the least recently used chunks until the capacity limit is satisfied.
- The SSD tier is implicit and does not need to be returned.

Return a tuple `(resident, total_savings)` where `resident` is the list of final CPU-resident chunk ids in increasing id order, and `total_savings` is the sum of `savings[id]` over those resident chunks.

Assume all chunks in `trace` exist in `sizes` and `savings`.

## Example

```python
trace = [1, 2, 1, 3]
sizes = {1: 4, 2: 6, 3: 5}
savings = {1: 10, 2: 7, 3: 8}

resident, total = tier_lru(trace, sizes, savings, 10)
# resident == [1, 3]
# total == 18
```

## What the gate checks

The gate runs the implementation against several access traces and compares the returned resident set and token savings with an LRU oracle. The oracle independently simulates CPU residency, recency updates, and byte-capacity evictions, then computes the final savings value.

A solution only passes if it matches the oracle exactly.
