## Context

An Access Pattern Cache (APC) holds a fixed-capacity pool $\mathcal{P}$ of at most $C$ blocks, each identified by an integer chain hash.  When every slot is occupied and a new block arrives, the cache must choose a victim to evict.  The **Least-Recently-Used (LRU)** policy picks the block whose last access time is smallest.

Let $\text{recency}(b)$ be the step index at which block $b$ was last touched, and let $r_1, r_2, \ldots, r_n$ be the request sequence.  At step $t$:

- **Hit** ($r_t \in \mathcal{P}$): increment the hit counter; update $\text{recency}(r_t) \leftarrow t$.
- **Miss** ($r_t \notin \mathcal{P}$): if $|\mathcal{P}| = C$, evict the victim $b^{*} = \arg\min_{b \in \mathcal{P}} \text{recency}(b)$ and record its hash; then insert $r_t$ with $\text{recency}(r_t) \leftarrow t$.

The hit rate over $n$ requests is $H / n$ where $H$ is the total hit count.

A practical implementation can exploit the fact that Python $\ge 3.7$ dicts preserve insertion order: removing a key and re-inserting it moves it to the end, giving $O(1)$ amortised recency updates.

## Task

Implement `lru_apc(requests, pool_capacity)`:

```python
def lru_apc(requests, pool_capacity):
    ...
```

**Parameters:**

- `requests`: a list of `int` — chain hashes requested in order.
- `pool_capacity`: a positive `int` $C$ — maximum number of blocks the pool can hold.

**Returns:**

A tuple `(hit_count, evicted_order)` where

- `hit_count` (`int`): total number of cache hits.
- `evicted_order` (`list[int]`): chain hashes of evicted blocks, in the order they were evicted.

The pool starts empty.

## Example

```python
requests = [1, 2, 3, 1, 2, 4]
pool_capacity = 3

hit_count, evicted_order = lru_apc(requests, pool_capacity)
# hit_count = 2   (re-requests of 1 and 2)
# evicted_order = [3]   (3 was the LRU victim when 4 arrived)
```

| Step | Request | Pool (oldest → newest) | Hit? | Evicted |
|------|---------|------------------------|------|---------|
| 1 | 1 | $\{1\}$ | miss | — |
| 2 | 2 | $\{1, 2\}$ | miss | — |
| 3 | 3 | $\{1, 2, 3\}$ | miss | — |
| 4 | 1 | $\{2, 3, 1\}$ | hit | — |
| 5 | 2 | $\{3, 1, 2\}$ | hit | — |
| 6 | 4 | $\{1, 2, 4\}$ | miss | 3 |

## What the gate checks

The gate metric is `exact_match` (threshold 1.0).  Ten test cases are evaluated:

1. Basic hit-miss mix (capacity 3).
2. Long miss-only eviction chain (capacity 3).
3. All hits after warm-up (capacity 3).
4. Capacity-1 pool with alternating access (capacity 1).
5. Empty request list (capacity 3).
6. Single-block repeated access (capacity 5).
7. Interleaved hit/miss with evictions (capacity 3).
8. Three full passes over 20 unique keys (capacity 5).
9. Pattern that keeps one hot key alive while others churn (capacity 3).
10. Small capacity with back-to-back evictions (capacity 2).

Both `hit_count` and `evicted_order` must match a reference oracle (built from `collections.OrderedDict`) exactly on every case.  Any mismatch sets the gate to 0.
