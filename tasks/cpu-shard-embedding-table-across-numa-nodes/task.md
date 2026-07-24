## Context

A CPU with $n_{\text{nodes}}$ NUMA nodes has memory divided across nodes. Accessing local
memory is cheap, while remote memory across nodes adds latency.

An embedding table $E \in \mathbb{R}^{N \times D}$ stores $N$ embedding vectors of
dimension $D$. When queries request a batch of indices $i_1, \dots, i_B$, the goal
is to place rows of $E$ across NUMA nodes such that most lookups read local
memory for their assigned node.

Model each access as a memory address: the table is laid out row-major, each
element is `float64` (8 bytes). Assume a deterministic cache with line size
`line_bytes` and fixed sets/ways. A simple simulator counts cache misses across
an access trace. Each node may have its own region of contiguous addresses.

If each process handles a batch of indices meant for its node, a good sharding
minimizes *remote* accesses—those whose chosen addresses belong to other nodes’
regions.

## Task

Implement

```python
def shard_embedding_table(num_embeddings: int, dim: int, num_nodes: int) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) index ranges (in rows) assigning the embedding
    table rows to each NUMA node contiguously. Cover all rows [0, num_embeddings),
    with approximately equal local shard sizes differing by at most one row.
    """
```

Rows $[s_j, e_j)$ belong to node $j$. For example, with 10 000 embeddings, 4 nodes,
ranges could be roughly `[(0,2500), (2500,5000), (5000,7500), (7500,10000)]`.

Your function need not model caches, but the grader will. Deterministically divide
the table to balance shards.

## Example

```python
>>> shard_embedding_table(10, 3, 4)
[(0, 3), (3, 6), (6, 8), (8, 10)]
```

Each tuple’s second element is exclusive. The small example divides 10 rows across
4 nodes as evenly as possible.

## What the gate checks

The grader synthesizes synthetic queries routed per node. For each candidate
sharding, it simulates access traces with `arena.cachesim.simulate`. It computes
the reference’s cache miss count via the same model and expects equality.

Metric: `"exact_match" == 1.0` when your sharding yields an identical modeled miss
count under the simulator. The check is deterministic and hardware-independent,
using synthetic traces — no wall-clock timing. Only a contiguous equal partition
passes.
