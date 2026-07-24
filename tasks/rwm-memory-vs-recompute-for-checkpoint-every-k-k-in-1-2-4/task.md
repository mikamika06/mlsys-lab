## Context

Activation checkpointing trades memory for additional computation during training.

Consider a sequential network with $L$ forward layers. Each layer produces an
activation tensor occupying $B$ bytes. Instead of storing every intermediate
activation, a checkpointing strategy stores only selected boundary activations.
During the backward pass, missing activations are recomputed by running forward
layers again from the nearest stored checkpoint.

For a checkpoint interval $k$, the implementation stores the activation at the
input and after every $k$ layers. The number of stored activation tensors is

$$
1 + \left\lceil \frac{L}{k} \right\rceil .
$$

The stored activation memory is therefore

$$
M_k = B \left(1 + \left\lceil \frac{L}{k} \right\rceil \right).
$$

For each checkpoint segment, the backward pass must replay the layers inside the
segment to regenerate discarded activations. The total extra forward-layer work
is the sum of the segment lengths excluding the checkpoint boundary layers.

This task models the memory/compute frontier used by checkpointing systems. The
possible checkpoint intervals are fixed to $k \in \{1,2,4\}$.

## Task

Implement `checkpoint_curve(L, activation_bytes)`:

```python
def checkpoint_curve(L: int, activation_bytes: int) -> list:
    ...
```

Return a list containing one entry for each $k$ in `[1, 2, 4]`. Each entry must
be a list of the form:

```python
[k, stored_activation_bytes, extra_forward_layers]
```

where:

- `stored_activation_bytes` is the number of bytes occupied by stored checkpoint
  activations.
- `extra_forward_layers` is the number of additional forward layer executions
  required during backward recomputation.

Assume $L$ is a positive integer and `activation_bytes` is a positive integer.

## Example

```python
checkpoint_curve(5, 1024)
```

returns:

```python
[
    [1, 6144, 0],
    [2, 4096, 3],
    [4, 3072, 5]
]
```

## What the gate checks

The gate constructs several layer counts and activation sizes. It computes the
checkpoint memory and recomputation curve with an independent oracle algorithm
and compares every returned $k$ entry exactly.

A solution that uses an incorrect checkpoint count or an incorrect recomputation
formula will fail even if it handles simple cases.
