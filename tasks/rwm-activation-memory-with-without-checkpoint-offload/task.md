## Context

Transformer training stores intermediate activations so that backpropagation can
reuse them. If every layer keeps its output, the activation memory grows with
the number of layers.

For a transformer with depth $L$, sequence length $S$, hidden size $H$, and
activation dtype size $b$ bytes, storing every layer output requires

$$
M_{\mathrm{full}} = L \cdot S \cdot H \cdot b .
$$

Production training systems often use activation checkpointing. Instead of
keeping every layer output, they keep only checkpoint boundaries and recompute
the missing activations during the backward pass. A simple square-root
checkpoint schedule keeps approximately $\lceil\sqrt{L}\rceil$ boundaries:

$$
M_{\mathrm{checkpoint}} =
\lceil\sqrt{L}\rceil \cdot S \cdot H \cdot b .
$$

Activation offload moves saved activations out of GPU memory, so the GPU
activation footprint is approximately zero:

$$
M_{\mathrm{offload}} = 0 .
$$

## Task

Implement `activation_memory`:

```python
def activation_memory(depth: int, seq: int, hidden: int, dtype_bytes: int) -> dict:
    ...
```

Return a dictionary with exactly these keys:

- `"full_store"`: bytes used when all layer activations remain stored.
- `"checkpoint"`: bytes used when square-root spaced checkpoint boundaries are
  stored.
- `"offload"`: bytes remaining on GPU when all activations are offloaded.

The checkpoint boundary count is `ceil(sqrt(depth))`. Use integer arithmetic for
the returned byte counts.

## Example

```python
result = activation_memory(100, 512, 1024, 2)

# {
#   "full_store": 104857600,
#   "checkpoint": 10485760,
#   "offload": 0
# }
```

## What the gate checks

The gate generates several depth, sequence length, hidden size, and dtype
configurations. It computes the expected values using an independent oracle
implementation of the memory formulas and requires an exact dictionary match.
