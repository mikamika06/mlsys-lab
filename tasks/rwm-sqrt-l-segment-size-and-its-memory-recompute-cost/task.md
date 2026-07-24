## Context

Activation checkpointing reduces training memory by storing only selected intermediate activations and recomputing missing activations during the backward pass.

For a network with $L$ uniform layers, divide the layers into equal segments of length $s$. The classic square-root checkpointing approximation chooses

$$
s \approx \sqrt{L}.
$$

The number of stored segment boundary activations is approximately

$$
2s,
$$

because the algorithm keeps the segment checkpoints and the currently recomputed working activations. The recomputation overhead for the square-root schedule is one additional full forward traversal:

$$
\text{extra forward work} \approx L.
$$

This task models the planning step used by a training runtime before execution. The runtime needs the segment size, estimated stored activations, and recomputation count.

## Task

Implement `checkpoint_cost(L)`:

```python
def checkpoint_cost(L: int) -> tuple[int, int, int]:
    ...
```

The function receives the number of uniform layers $L$ and returns:

1. `segment_size`: the chosen segment size, computed as `round(sqrt(L))`.
2. `stored_activations`: the estimated number of activations kept in memory, computed as $2 \times \text{segment\_size}$.
3. `extra_forward`: the number of extra forward layer evaluations required, computed as $L$.

Use integer arithmetic for the returned values.

## Example

```python
segment_size, stored, extra = checkpoint_cost(100)

# segment_size == 10
# stored == 20
# extra == 100
```

## What the gate checks

The gate builds an oracle implementation of the square-root checkpointing formula and compares all returned integer values for several layer counts.

The returned tuple must exactly match the oracle output:
$$(\operatorname{round}(\sqrt{L}), 2\operatorname{round}(\sqrt{L}), L).$$
