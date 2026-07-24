## Context

ZeRO-2 partitions optimizer state and gradients across data-parallel ranks. After each rank computes a full gradient tensor, the reduce-scatter operation first performs an elementwise sum across ranks and then assigns each contiguous shard to its owning rank.

For $R$ ranks, let the local gradients be

$$G^{(0)}, G^{(1)}, \dots, G^{(R-1)} \in \mathbb{R}^{N}.$$

The reduced gradient is

$$G = \sum_{r=0}^{R-1} G^{(r)}.$$

The ownership rule is that rank $r$ receives the shard containing indices assigned to that rank. With equal-sized shards, rank $r$ owns

$$G\left[\frac{rN}{R}:\frac{(r+1)N}{R}\right].$$

A common implementation bug is to split each local gradient before reduction or to send shards to the wrong rank. This leaves the values incorrect even though the tensor shapes may look valid.

## Task

Implement `reduce_scatter_owner(grads, world_size)`:

```python
def reduce_scatter_owner(grads: list[list[float]], world_size: int) -> list[list[float]]:
    ...
```

`grads` contains one full gradient vector per rank. All vectors have the same length, and the length is divisible by `world_size`.

Return a list of `world_size` gradient shards. Element `result[r]` must be the shard owned by rank $r$ after reduce-scatter.

The function must:
1. Sum gradients from all ranks elementwise.
2. Split the summed gradient into contiguous equal-sized shards.
3. Place each shard at the index of its owning rank.

## Example

```python
grads = [
    [1, 2, 3, 4],
    [10, 20, 30, 40],
]

reduce_scatter_owner(grads, 2)
# [
#   [11, 22],
#   [33, 44],
# ]
```

## What the gate checks

The gate builds several gradient sets and computes the expected ownership using a NumPy reduce-scatter oracle. The submitted function must exactly match the oracle output for every case.

The `exact_match` score is $1.0$ only when every returned shard has the correct owner and every value matches.
