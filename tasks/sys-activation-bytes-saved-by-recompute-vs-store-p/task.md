## Context

FlashAttention reduces activation memory in attention models by avoiding storage of
the full attention probability matrix $P$ during training. A standard attention
layer with sequence length $N$ and head dimension $d$ can materialize

$$
P = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right),
$$

where $P \in \mathbb{R}^{N \times N}$.

If $P$ is stored for the backward pass, the activation memory contribution is
quadratic:

$$
B_{\mathrm{store}} = N^2 \cdot s,
$$

where $s$ is the number of bytes per element. With recomputation, the backward
pass stores only smaller inputs such as $Q$, $K$, and $V$:

$$
B_{\mathrm{recompute}} = 3Nd \cdot s .
$$

The saved-memory ratio is

$$
r = \frac{B_{\mathrm{recompute}}}{B_{\mathrm{store}}}
= \frac{3d}{N}.
$$

For large sequence lengths $N \gg d$, recomputation uses much less activation
memory than storing $P$.

## Task

Implement `activation_bytes_saved(N, d, bytes_per_element)`:

```python
def activation_bytes_saved(N: int, d: int, bytes_per_element: int) -> tuple[int, int]:
    ...
```

Return a tuple:

1. The modeled activation bytes required when storing $P$.
2. The modeled activation bytes required when recomputing attention activations.

Use integer arithmetic and return `(store_bytes, recompute_bytes)`.

## Example

```python
store, recompute = activation_bytes_saved(4096, 64, 2)

# store is the bytes for a 4096 x 4096 fp16 matrix
# recompute is the bytes for Q, K, and V stored as fp16 matrices
```

## What the gate checks

The gate computes the expected byte counts using an independent NumPy-based
oracle that models the tensor shapes and dtype item size. The returned values
must match the oracle exactly. The reported `size_ratio` is `1.0` only when the
implementation produces the correct store-versus-recompute byte model.
