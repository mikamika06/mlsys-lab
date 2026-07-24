## Context

A reduce-scatter collective takes data from every rank, reduces matching positions,
and distributes one reduced shard to each rank. Assume there are $p$ ranks and each
rank contributes a flat buffer split into $p$ equal chunks.

For rank $r$, chunk $c$ contains the values that belong to output rank $c$.
The reduction for a chunk is the elementwise sum:

$$
S_c = \sum_{r=0}^{p-1} X_{r,c},
$$

where $X_{r,c}$ is chunk $c$ from rank $r$. The final owner of chunk $c$ is rank
$c$.

A common implementation bug is an offset error when slicing a rank buffer. The
correct chunk start is

$$
\mathrm{start} = c \cdot k,
$$

where $k$ is the chunk length. Using $(c+1) \cdot k$ shifts every shard and
causes the wrong rank to receive the reduced data.

## Task

Fix `reduce_scatter_chunks` so that it returns the correctly reduced shard for
each output rank.

The function contract is:

```python
def reduce_scatter_chunks(buffers, world_size):
    ...
```

`buffers` is a list of `world_size` flat Python lists. Each list has length
`world_size * chunk_size`. The function returns a list of `world_size` lists.
The item at index $r$ must be the reduced chunk owned by rank $r$.

The existing implementation contains an offset bug. Modify it rather than
replacing the interface.

## Example

```python
buffers = [
    [1, 2, 10, 20],
    [3, 4, 30, 40],
]
reduce_scatter_chunks(buffers, 2)

# rank 0 owns chunk 0: [1, 2] + [3, 4]
# rank 1 owns chunk 1: [10, 20] + [30, 40]
# result:
# [[4, 6], [40, 60]]
```

## What the gate checks

The gate computes the expected reduce-scatter result using an independent reference
implementation of the chunk reduction algorithm. It compares the repaired
function output against that result on multiple buffer layouts.

The `exact_match` metric must equal $1.0$. Any chunk offset shift or incorrect
ownership mapping fails the gate.
