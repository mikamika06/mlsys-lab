## Context

A prefix-caching radix tree stores every distinct token *position* it has
ever seen exactly once: if two sequences share a prefix, that prefix's
nodes are shared (one copy), and only where they diverge do separate
branches appear. A node's identity is the full path of tokens from the
root down to it — two sequences create the same node for a shared prefix
position, but two different nodes wherever they differ. The tree's total
token count is simply its number of distinct nodes.

When the tree exceeds a token `capacity`, something has to be evicted.
The safe rule: only a **leaf** (a node with no live children) may ever be
evicted — evicting a node that still has children would silently corrupt
every sequence still relying on that shared trunk. Among all current
leaves, evict the **least-recently-touched** one first; removing it may
turn its parent into a new leaf, which becomes eligible next.

"Touched" applies to reads as well as writes: a node counts as recently
used whenever **any** operation walks through it, whether inserting new
tokens or looking an existing sequence back up. To make "least recently
touched" unambiguous, every single node-touch (not every operation) gets
its own tick of a strictly increasing global clock, in path order (the
node closest to the root is ticked first): walking a sequence's `L`-token
path touches (and re-stamps) all `L` nodes along it, one tick each,
whether or not that node already existed.

## Task

Implement `radix_lru_cache`:

```python
def radix_lru_cache(
    trace: list[tuple[str, list[int]]],
    capacity: int,
    followup: list[list[int]],
) -> tuple[int, float]:
    ...
```

- `trace`: a list of `("insert", seq)` or `("query", seq)` operations,
  processed strictly in order. **Both** op kinds do exactly the same
  thing to the tree: walk `seq`'s prefixes root-to-leaf, creating any
  node that doesn't exist yet, and tick the global clock once per node
  touched (existing or new), stamping that node's `last_touch` with the
  new clock value.
- `capacity`: positive int, the max number of tree nodes (tokens)
  allowed at any time.
- `followup`: sequences to check **after** the whole trace finishes
  (this does not itself touch or modify the tree).

After **every** operation in `trace`, while the tree has more than
`capacity` nodes, repeatedly evict the current leaf with the smallest
`last_touch` value (one node per eviction step) until it's back at or
under `capacity`.

Return `(evicted_token_total, hit_rate)`:

- `evicted_token_total`: total number of nodes evicted across the entire
  trace.
- `hit_rate`: the fraction of `followup` sequences whose **entire**
  path (every token, root to tip) is still present in the tree after the
  trace finishes.

## Example

```python
trace = [
    ("insert", [1, 2, 3]),
    ("insert", [1, 2, 4]),
    ("insert", [1, 5]),
]
radix_lru_cache(trace, capacity=4, followup=[[1, 2, 3], [1, 2, 4], [1, 5]])
# After the first two inserts: nodes (1,) (1,2) (1,2,3) (1,2,4) -- 4 nodes, at capacity.
# Inserting [1,5] touches (1,) again and creates (1,5): 5 nodes, 1 over
#   capacity. The two deepest leaves are (1,2,3) and (1,2,4); (1,2,3) has
#   the older last_touch (it wasn't touched by the [1,2,4] insert), so
#   it's evicted. evicted_token_total = 1.
# Final tree has (1,2,3) gone but (1,2,4) and (1,5) intact ->
# hit_rate = 2/3 (only [1,2,3] misses).
```

## What the gate checks

The grader builds several `(trace, capacity, followup)` scenarios —
hand-built small traces exercising simultaneous multi-node eviction
(inserting a long brand-new branch under a tight capacity, forcing
several evictions in one step), a capacity large enough that nothing is
ever evicted, a `query` that refreshes a node's recency and changes which
leaf gets evicted next, and a longer trace built from a seeded NumPy
generator over a small shared vocabulary (so branches genuinely overlap)
— and computes the reference `(evicted_token_total, hit_rate)`
independently in Python, following the exact tick-per-node-touch and
evict-least-recently-touched-leaf rules above, never calling your
function or hardcoding an expected value.

`exact_match` is the fraction of scenarios where **both** returned values
match the oracle's exactly, and the gate requires `1.0`. Evicting by
whole-operation recency instead of per-node tick order, evicting a node
that still has a live child, not re-checking capacity after every single
eviction (stopping one node too early or too late), or forgetting that
`query` refreshes recency just like `insert` will all diverge from the
oracle on at least one scenario.
