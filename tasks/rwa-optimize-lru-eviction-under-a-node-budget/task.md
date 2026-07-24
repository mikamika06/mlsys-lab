## Context

A bounded tree cache stores recently used objects and must decide which leaf
nodes to remove when the node budget is exceeded. A least-recently-used (LRU)
policy keeps the leaves with the highest future reuse probability by removing
the leaf whose last access time is smallest.

The cache receives a sequence of operations. An insertion adds a leaf if the
key is absent. A query is a cache hit when the key currently exists in the
tree. Every successful query refreshes the key's recency timestamp.

If the number of cached leaves exceeds the budget $B$, the cache repeatedly
evicts the least-recently-used leaf until the constraint

$$|C| \leq B$$

is satisfied, where $C$ is the set of cached keys.

## Task

Implement `optimize_lru_cache(ops, budget)`:

```python
def optimize_lru_cache(ops: list[tuple[str, int]], budget: int) -> tuple[list[int], int]:
    ...
```

The input `ops` contains operations of the form `("insert", key)` and
`("query", key)`. Keys are non-negative integers. The function must simulate a
bounded radix-tree cache using LRU leaf eviction.

Return a tuple:

1. A list of the keys remaining in the cache after all operations, sorted in
   ascending order.
2. The cumulative number of successful query hits.

The implementation should maintain recency correctly. When multiple evictions
are required, remove the least-recently-used leaves one at a time.

## Example

```python
ops = [
    ("insert", 10),
    ("insert", 20),
    ("query", 10),
    ("insert", 30),
]

result = optimize_lru_cache(ops, 2)
# ([10, 30], 1)
```

The key `20` is removed because `10` was refreshed by the query before the
third insertion.

## What the gate checks

The gate replays several operation streams through an independent radix-tree
LRU oracle. It compares the returned cached key set and cumulative hit count
against the oracle output.

The returned tuple must exactly match the oracle result. The oracle computes
the result from the eviction algorithm itself rather than using stored expected
answers.
